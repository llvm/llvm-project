(*===---------------------------------------------------------------------===
 * Parser
 *===---------------------------------------------------------------------===*)

(* binop_precedence - This holds the precedence for each binary operator that is
 * defined *)
let binop_precedence:(char, int) Hashtbl.t = Hashtbl.create 10

(* precedence - Get the precedence of the pending binary operator token. *)
let precedence c = try Hashtbl.find binop_precedence c with Not_found -> -1

(* primary
 *   ::= identifier
 *   ::= numberexpr
 *   ::= parenexpr
 *   ::= ifexpr
 *   ::= forexpr
 *   ::= varexpr *)
let rec parse_primary = parser
  (* numberexpr ::= number *)
  | [< 'Token.Number n >] -> Ast.Number n

  (* parenexpr ::= '(' expression ')' *)
  | [< 'Token.Kwd '('; e=parse_expr; 'Token.Kwd ')' ?? "expected ')'" >] -> e

  (* identifierexpr
   *   ::= identifier
   *   ::= identifier '(' argumentexpr ')' *)
  | [< 'Token.Ident id; stream >] ->
      let rec parse_args accumulator = parser
        | [< e=parse_expr; stream >] ->
            begin parser
              | [< 'Token.Kwd ','; e=parse_args (e :: accumulator) >] -> e
              | [< >] -> e :: accumulator
            end stream
        | [< >] -> accumulator
      in
      let rec parse_ident id = parser
        (* Call. *)
        | [< 'Token.Kwd '(';
             args=parse_args [];
             'Token.Kwd ')' ?? "expected ')'">] ->
            Ast.Call (id, Array.of_list (List.rev args))

        (* Simple variable ref. *)
        | [< >] -> Ast.Variable id
      in
      parse_ident id stream

  (* ifexpr ::= 'if' expr 'then' expr 'else' expr *)
  | [< 'Token.If; c=parse_expr;
       'Token.Then ?? "expected 'then'"; t=parse_expr;
       'Token.Else ?? "expected 'else'"; e=parse_expr >] ->
      Ast.If (c, t, e)

  (* forexpr
        ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression *)
  | [< 'Token.For;
       'Token.Ident id ?? "expected identifier after for";
       'Token.Kwd '=' ?? "expected '=' after for";
       stream >] ->
      begin parser
        | [<
             start=parse_expr;
             'Token.Kwd ',' ?? "expected ',' after for";
             end_=parse_expr;
             stream >] ->
            let step =
              begin parser
              | [< 'Token.Kwd ','; step=parse_expr >] -> Some step
              | [< >] -> None
              end stream
            in
            begin parser
            | [< 'Token.In; body=parse_expr >] ->
                Ast.For (id, start, end_, step, body)
            | [< >] ->
                raise (Stream.Error "expected 'in' after for")
            end stream
        | [< >] ->
            raise (Stream.Error "expected '=' after for")
      end stream

  (* varexpr
   *   ::= 'var' identifier ('=' expression?
   *             (',' identifier ('=' expression)?)* 'in' expression *)
  | [< 'Token.Var;
       (* At least one variable name is required. *)
       'Token.Ident id ?? "expected identifier after var";
       init=parse_var_init;
       var_names=parse_var_names [(id, init)];
       (* At this point, we have to have 'in'. *)
       'Token.In ?? "expected 'in' keyword after 'var'";
       body=parse_expr >] ->
      Ast.Var (Array.of_list (List.rev var_names), body)

  | [< >] -> raise (Stream.Error "unknown token when expecting an expression.")

(* unary
 *   ::= primary
 *   ::= '!' unary *)
and parse_unary = parser
  (* If this is a unary operator, read it. *)
  | [< 'Token.Kwd op when op != '(' && op != ')'; operand=parse_expr >] ->
      Ast.Unary (op, operand)

  (* If the current token is not an operator, it must be a primary expr. *)
  | [< stream >] -> parse_primary stream

(* binoprhs
 *   ::= ('+' primary)* *)
and parse_bin_rhs expr_prec lhs stream =
  match Stream.peek stream with
  (* If this is a binop, find its precedence. *)
  | Some (Token.Kwd c) when Hashtbl.mem binop_precedence c ->
      let token_prec = precedence c in

      (* If this is a binop that binds at least as tightly as the current binop,
       * consume it, otherwise we are done. *)
      if token_prec < expr_prec then lhs else begin
        (* Eat the binop. *)
        Stream.junk stream;

        (* Parse the primary expression after the binary operator. *)
        let rhs = parse_unary stream in

        (* Okay, we know this is a binop. *)
        let rhs =
          match Stream.peek stream with
          | Some (Token.Kwd c2) ->
              (* If BinOp binds less tightly with rhs than the operator after
               * rhs, let the pending operator take rhs as its lhs. *)
              let next_prec = precedence c2 in
              if token_prec < next_prec
              then parse_bin_rhs (token_prec + 1) rhs stream
              else rhs
          | _ -> rhs
        in

        (* Merge lhs/rhs. *)
        let lhs = Ast.Binary (c, lhs, rhs) in
        parse_bin_rhs expr_prec lhs stream
      end
  | _ -> lhs

and parse_var_init = parser
  (* read in the optional initializer. *)
  | [< 'Token.Kwd '='; e=parse_expr >] -> Some e
  | [< >] -> None

and parse_var_names accumulator = parser
  | [< 'Token.Kwd ',';
       'Token.Ident id ?? "expected identifier list after var";
       init=parse_var_init;
       e=parse_var_names ((id, init) :: accumulator) >] -> e
  | [< >] -> accumulator

(* expression
 *   ::= primary binoprhs *)
and parse_expr = parser
  | [< lhs=parse_unary; stream >] -> parse_bin_rhs 0 lhs stream

(* prototype
 *   ::= id '(' id* ')'
 *   ::= binary LETTER number? (id, id)
 *   ::= unary LETTER number? (id) *)
let parse_prototype =
  let rec parse_args accumulator = parser
    | [< 'Token.Ident id; e=parse_args (id::accumulator) >] -> e
    | [< >] -> accumulator
  in
  let parse_operator = parser
    | [< 'Token.Unary >] -> "unary", 1
    | [< 'Token.Binary >] -> "binary", 2
  in
  let parse_binary_precedence = parser
    | [< 'Token.Number n >] -> int_of_float n
    | [< >] -> 30
  in
  parser
  | [< 'Token.Ident id;
       'Token.Kwd '(' ?? "expected '(' in prototype";
       args=parse_args [];
       'Token.Kwd ')' ?? "expected ')' in prototype" >] ->
      (* success. *)
      Ast.Prototype (id, Array.of_list (List.rev args))
  | [< (prefix, kind)=parse_operator;
       'Token.Kwd op ?? "expected an operator";
       (* Read the precedence if present. *)
       binary_precedence=parse_binary_precedence;
       'Token.Kwd '(' ?? "expected '(' in prototype";
        args=parse_args [];
       'Token.Kwd ')' ?? "expected ')' in prototype" >] ->
      let name = prefix ^ (String.make 1 op) in
      let args = Array.of_list (List.rev args) in

      (* Verify right number of arguments for operator. *)
      if Array.length args != kind
      then raise (Stream.Error "invalid number of operands for operator")
      else
        if kind == 1 then
          Ast.Prototype (name, args)
        else
          Ast.BinOpPrototype (name, args, binary_precedence)
  | [< >] ->
      raise (Stream.Error "expected function name in prototype")

(* definition ::= 'def' prototype expression *)
let parse_definition = parser
  | [< 'Token.Def; p=parse_prototype; e=parse_expr >] ->
      Ast.Function (p, e)

(* toplevelexpr ::= expression *)
let parse_toplevel = parser
  | [< e=parse_expr >] ->
      (* Make an anonymous proto. *)
      Ast.Function (Ast.Prototype ("", [||]), e)

(*  external ::= 'extern' prototype *)
let parse_extern = parser
  | [< 'Token.Extern; e=parse_prototype >] -> e
