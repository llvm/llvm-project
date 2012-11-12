(*===----------------------------------------------------------------------===
 * Code Generation
 *===----------------------------------------------------------------------===*)

open Llvm

exception Error of string

let context = global_context ()
let the_module = create_module context "my cool jit"
let builder = builder context
let named_values:(string, llvalue) Hashtbl.t = Hashtbl.create 10
let double_type = double_type context

(* Create an alloca instruction in the entry block of the function. This
 * is used for mutable variables etc. *)
let create_entry_block_alloca the_function var_name =
  let builder = builder_at context (instr_begin (entry_block the_function)) in
  build_alloca double_type var_name builder

let rec codegen_expr = function
  | Ast.Number n -> const_float double_type n
  | Ast.Variable name ->
      let v = try Hashtbl.find named_values name with
        | Not_found -> raise (Error "unknown variable name")
      in
      (* Load the value. *)
      build_load v name builder
  | Ast.Unary (op, operand) ->
      let operand = codegen_expr operand in
      let callee = "unary" ^ (String.make 1 op) in
      let callee =
        match lookup_function callee the_module with
        | Some callee -> callee
        | None -> raise (Error "unknown unary operator")
      in
      build_call callee [|operand|] "unop" builder
  | Ast.Binary (op, lhs, rhs) ->
      begin match op with
      | '=' ->
          (* Special case '=' because we don't want to emit the LHS as an
           * expression. *)
          let name =
            match lhs with
            | Ast.Variable name -> name
            | _ -> raise (Error "destination of '=' must be a variable")
          in

          (* Codegen the rhs. *)
          let val_ = codegen_expr rhs in

          (* Lookup the name. *)
          let variable = try Hashtbl.find named_values name with
          | Not_found -> raise (Error "unknown variable name")
          in
          ignore(build_store val_ variable builder);
          val_
      | _ ->
          let lhs_val = codegen_expr lhs in
          let rhs_val = codegen_expr rhs in
          begin
            match op with
            | '+' -> build_fadd lhs_val rhs_val "addtmp" builder
            | '-' -> build_fsub lhs_val rhs_val "subtmp" builder
            | '*' -> build_fmul lhs_val rhs_val "multmp" builder
            | '<' ->
                (* Convert bool 0/1 to double 0.0 or 1.0 *)
                let i = build_fcmp Fcmp.Ult lhs_val rhs_val "cmptmp" builder in
                build_uitofp i double_type "booltmp" builder
            | _ ->
                (* If it wasn't a builtin binary operator, it must be a user defined
                 * one. Emit a call to it. *)
                let callee = "binary" ^ (String.make 1 op) in
                let callee =
                  match lookup_function callee the_module with
                  | Some callee -> callee
                  | None -> raise (Error "binary operator not found!")
                in
                build_call callee [|lhs_val; rhs_val|] "binop" builder
          end
      end
  | Ast.Call (callee, args) ->
      (* Look up the name in the module table. *)
      let callee =
        match lookup_function callee the_module with
        | Some callee -> callee
        | None -> raise (Error "unknown function referenced")
      in
      let params = params callee in

      (* If argument mismatch error. *)
      if Array.length params == Array.length args then () else
        raise (Error "incorrect # arguments passed");
      let args = Array.map codegen_expr args in
      build_call callee args "calltmp" builder
  | Ast.If (cond, then_, else_) ->
      let cond = codegen_expr cond in

      (* Convert condition to a bool by comparing equal to 0.0 *)
      let zero = const_float double_type 0.0 in
      let cond_val = build_fcmp Fcmp.One cond zero "ifcond" builder in

      (* Grab the first block so that we might later add the conditional branch
       * to it at the end of the function. *)
      let start_bb = insertion_block builder in
      let the_function = block_parent start_bb in

      let then_bb = append_block context "then" the_function in

      (* Emit 'then' value. *)
      position_at_end then_bb builder;
      let then_val = codegen_expr then_ in

      (* Codegen of 'then' can change the current block, update then_bb for the
       * phi. We create a new name because one is used for the phi node, and the
       * other is used for the conditional branch. *)
      let new_then_bb = insertion_block builder in

      (* Emit 'else' value. *)
      let else_bb = append_block context "else" the_function in
      position_at_end else_bb builder;
      let else_val = codegen_expr else_ in

      (* Codegen of 'else' can change the current block, update else_bb for the
       * phi. *)
      let new_else_bb = insertion_block builder in

      (* Emit merge block. *)
      let merge_bb = append_block context "ifcont" the_function in
      position_at_end merge_bb builder;
      let incoming = [(then_val, new_then_bb); (else_val, new_else_bb)] in
      let phi = build_phi incoming "iftmp" builder in

      (* Return to the start block to add the conditional branch. *)
      position_at_end start_bb builder;
      ignore (build_cond_br cond_val then_bb else_bb builder);

      (* Set a unconditional branch at the end of the 'then' block and the
       * 'else' block to the 'merge' block. *)
      position_at_end new_then_bb builder; ignore (build_br merge_bb builder);
      position_at_end new_else_bb builder; ignore (build_br merge_bb builder);

      (* Finally, set the builder to the end of the merge block. *)
      position_at_end merge_bb builder;

      phi
  | Ast.For (var_name, start, end_, step, body) ->
      (* Output this as:
       *   var = alloca double
       *   ...
       *   start = startexpr
       *   store start -> var
       *   goto loop
       * loop:
       *   ...
       *   bodyexpr
       *   ...
       * loopend:
       *   step = stepexpr
       *   endcond = endexpr
       *
       *   curvar = load var
       *   nextvar = curvar + step
       *   store nextvar -> var
       *   br endcond, loop, endloop
       * outloop: *)

      let the_function = block_parent (insertion_block builder) in

      (* Create an alloca for the variable in the entry block. *)
      let alloca = create_entry_block_alloca the_function var_name in

      (* Emit the start code first, without 'variable' in scope. *)
      let start_val = codegen_expr start in

      (* Store the value into the alloca. *)
      ignore(build_store start_val alloca builder);

      (* Make the new basic block for the loop header, inserting after current
       * block. *)
      let loop_bb = append_block context "loop" the_function in

      (* Insert an explicit fall through from the current block to the
       * loop_bb. *)
      ignore (build_br loop_bb builder);

      (* Start insertion in loop_bb. *)
      position_at_end loop_bb builder;

      (* Within the loop, the variable is defined equal to the PHI node. If it
       * shadows an existing variable, we have to restore it, so save it
       * now. *)
      let old_val =
        try Some (Hashtbl.find named_values var_name) with Not_found -> None
      in
      Hashtbl.add named_values var_name alloca;

      (* Emit the body of the loop.  This, like any other expr, can change the
       * current BB.  Note that we ignore the value computed by the body, but
       * don't allow an error *)
      ignore (codegen_expr body);

      (* Emit the step value. *)
      let step_val =
        match step with
        | Some step -> codegen_expr step
        (* If not specified, use 1.0. *)
        | None -> const_float double_type 1.0
      in

      (* Compute the end condition. *)
      let end_cond = codegen_expr end_ in

      (* Reload, increment, and restore the alloca. This handles the case where
       * the body of the loop mutates the variable. *)
      let cur_var = build_load alloca var_name builder in
      let next_var = build_add cur_var step_val "nextvar" builder in
      ignore(build_store next_var alloca builder);

      (* Convert condition to a bool by comparing equal to 0.0. *)
      let zero = const_float double_type 0.0 in
      let end_cond = build_fcmp Fcmp.One end_cond zero "loopcond" builder in

      (* Create the "after loop" block and insert it. *)
      let after_bb = append_block context "afterloop" the_function in

      (* Insert the conditional branch into the end of loop_end_bb. *)
      ignore (build_cond_br end_cond loop_bb after_bb builder);

      (* Any new code will be inserted in after_bb. *)
      position_at_end after_bb builder;

      (* Restore the unshadowed variable. *)
      begin match old_val with
      | Some old_val -> Hashtbl.add named_values var_name old_val
      | None -> ()
      end;

      (* for expr always returns 0.0. *)
      const_null double_type
  | Ast.Var (var_names, body) ->
      let old_bindings = ref [] in

      let the_function = block_parent (insertion_block builder) in

      (* Register all variables and emit their initializer. *)
      Array.iter (fun (var_name, init) ->
        (* Emit the initializer before adding the variable to scope, this
         * prevents the initializer from referencing the variable itself, and
         * permits stuff like this:
         *   var a = 1 in
         *     var a = a in ...   # refers to outer 'a'. *)
        let init_val =
          match init with
          | Some init -> codegen_expr init
          (* If not specified, use 0.0. *)
          | None -> const_float double_type 0.0
        in

        let alloca = create_entry_block_alloca the_function var_name in
        ignore(build_store init_val alloca builder);

        (* Remember the old variable binding so that we can restore the binding
         * when we unrecurse. *)
        begin
          try
            let old_value = Hashtbl.find named_values var_name in
            old_bindings := (var_name, old_value) :: !old_bindings;
          with Not_found -> ()
        end;

        (* Remember this binding. *)
        Hashtbl.add named_values var_name alloca;
      ) var_names;

      (* Codegen the body, now that all vars are in scope. *)
      let body_val = codegen_expr body in

      (* Pop all our variables from scope. *)
      List.iter (fun (var_name, old_value) ->
        Hashtbl.add named_values var_name old_value
      ) !old_bindings;

      (* Return the body computation. *)
      body_val

let codegen_proto = function
  | Ast.Prototype (name, args) | Ast.BinOpPrototype (name, args, _) ->
      (* Make the function type: double(double,double) etc. *)
      let doubles = Array.make (Array.length args) double_type in
      let ft = function_type double_type doubles in
      let f =
        match lookup_function name the_module with
        | None -> declare_function name ft the_module

        (* If 'f' conflicted, there was already something named 'name'. If it
         * has a body, don't allow redefinition or reextern. *)
        | Some f ->
            (* If 'f' already has a body, reject this. *)
            if block_begin f <> At_end f then
              raise (Error "redefinition of function");

            (* If 'f' took a different number of arguments, reject. *)
            if element_type (type_of f) <> ft then
              raise (Error "redefinition of function with different # args");
            f
      in

      (* Set names for all arguments. *)
      Array.iteri (fun i a ->
        let n = args.(i) in
        set_value_name n a;
        Hashtbl.add named_values n a;
      ) (params f);
      f

(* Create an alloca for each argument and register the argument in the symbol
 * table so that references to it will succeed. *)
let create_argument_allocas the_function proto =
  let args = match proto with
    | Ast.Prototype (_, args) | Ast.BinOpPrototype (_, args, _) -> args
  in
  Array.iteri (fun i ai ->
    let var_name = args.(i) in
    (* Create an alloca for this variable. *)
    let alloca = create_entry_block_alloca the_function var_name in

    (* Store the initial value into the alloca. *)
    ignore(build_store ai alloca builder);

    (* Add arguments to variable symbol table. *)
    Hashtbl.add named_values var_name alloca;
  ) (params the_function)

let codegen_func the_fpm = function
  | Ast.Function (proto, body) ->
      Hashtbl.clear named_values;
      let the_function = codegen_proto proto in

      (* If this is an operator, install it. *)
      begin match proto with
      | Ast.BinOpPrototype (name, args, prec) ->
          let op = name.[String.length name - 1] in
          Hashtbl.add Parser.binop_precedence op prec;
      | _ -> ()
      end;

      (* Create a new basic block to start insertion into. *)
      let bb = append_block context "entry" the_function in
      position_at_end bb builder;

      try
        (* Add all arguments to the symbol table and create their allocas. *)
        create_argument_allocas the_function proto;

        let ret_val = codegen_expr body in

        (* Finish off the function. *)
        let _ = build_ret ret_val builder in

        (* Validate the generated code, checking for consistency. *)
        Llvm_analysis.assert_valid_function the_function;

        (* Optimize the function. *)
        let _ = PassManager.run_function the_function the_fpm in

        the_function
      with e ->
        delete_function the_function;
        raise e
