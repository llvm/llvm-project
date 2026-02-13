(*===----------------------------------------------------------------------===
 * Code Generation
 *===----------------------------------------------------------------------===*)

open Llvm

exception Error of string

let context = global_context ()
let the_module = create_module context "my cool jit"
let builder = builder context
let named_values:(string, llvalue) Hashtbl.t = Hashtbl.create 10
let function_types:(string, lltype) Hashtbl.t = Hashtbl.create 10
let double_type = double_type context

let rec codegen_expr = function
  | Ast.Number n -> const_float double_type n
  | Ast.Variable name ->
      (try Hashtbl.find named_values name with
        | Not_found -> raise (Error "unknown variable name"))
  | Ast.Binary (op, lhs, rhs) ->
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
        | _ -> raise (Error "invalid binary operator")
      end

  | Ast.Call (callee_name, args) ->
      let callee =
        match lookup_function callee_name the_module with
        | Some callee -> callee
        | None -> raise (Error "unknown function referenced")
      in
      let params = params callee in

      (* If argument mismatch error. *)
      if Array.length params == Array.length args then () else
        raise (Error "incorrect # arguments passed");

      let fnty =
        try Hashtbl.find function_types callee_name
        with Not_found ->
          raise (Error "unknown function type")
      in

      let args = Array.map codegen_expr args in
      build_call fnty callee args "calltmp" builder


let codegen_proto = function
  | Ast.Prototype (name, args) ->

      (* Make the function type: double(double,double) etc. *)
      let arg_types =
        Array.make (Array.length args) double_type
      in

      let ft = function_type double_type arg_types in
      Hashtbl.replace function_types name ft;

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
          set_value_name args.(i) a
        ) (params f);

      f

let codegen_func = function
  | Ast.Function (proto, body) ->
      Hashtbl.clear named_values;

      (* Get function name *)
      let name =
        match proto with
        | Ast.Prototype (n, _) -> n
      in

      (* Get or create prototype *)
      let the_function =
        match lookup_function name the_module with
        | Some f -> f
        | None -> codegen_proto proto
      in

      (* Reject redefinition *)
      if Array.length (basic_blocks the_function) <> 0 then
        raise (Error "redefinition of function");

      (* Create a new basic block to start insertion into. *)
      let bb = append_block context "entry" the_function in
      position_at_end bb builder;

      (* Register arguments *)
      Array.iter
        (fun a ->
          Hashtbl.add named_values (value_name a) a
        )
        (params the_function);

      try
        let ret_val = codegen_expr body in

        (* Finish off the function. *)
        let _ = build_ret ret_val builder in

        (* Validate the generated code, checking for consistency. *)
        Llvm_analysis.assert_valid_function the_function;

        the_function
      with e ->
        delete_function the_function;
        raise e

