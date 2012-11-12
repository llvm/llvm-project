(*===----------------------------------------------------------------------===
 * Top-Level parsing and JIT Driver
 *===----------------------------------------------------------------------===*)

open Llvm

(* top ::= definition | external | expression | ';' *)
let rec main_loop stream =
  match Stream.peek stream with
  | None -> ()

  (* ignore top-level semicolons. *)
  | Some (Token.Kwd ';') ->
      Stream.junk stream;
      main_loop stream

  | Some token ->
      begin
        try match token with
        | Token.Def ->
            let e = Parser.parse_definition stream in
            print_endline "parsed a function definition.";
            dump_value (Codegen.codegen_func e);
        | Token.Extern ->
            let e = Parser.parse_extern stream in
            print_endline "parsed an extern.";
            dump_value (Codegen.codegen_proto e);
        | _ ->
            (* Evaluate a top-level expression into an anonymous function. *)
            let e = Parser.parse_toplevel stream in
            print_endline "parsed a top-level expr";
            dump_value (Codegen.codegen_func e);
        with Stream.Error s | Codegen.Error s ->
          (* Skip token for error recovery. *)
          Stream.junk stream;
          print_endline s;
      end;
      print_string "ready> "; flush stdout;
      main_loop stream
