(* Tiny unit test framework - really just to help find which line is busted *)
let exit_status = ref 0

let suite_name = ref ""

let group_name = ref ""

let case_num = ref 0

let print_checkpoints = false

let group name =
  group_name := !suite_name ^ "/" ^ name;
  case_num := 0;
  if print_checkpoints then prerr_endline ("  " ^ name ^ "...")

let insist ?(exit_on_fail = false) cond =
  incr case_num;
  if not cond then exit_status := 10;
  ( match (print_checkpoints, cond) with
  | false, true -> ()
  | false, false ->
      prerr_endline
        ( "FAILED: " ^ !suite_name ^ "/" ^ !group_name ^ " #"
        ^ string_of_int !case_num )
  | true, true -> prerr_endline ("    " ^ string_of_int !case_num)
  | true, false -> prerr_endline ("    " ^ string_of_int !case_num ^ " FAIL") );
  if exit_on_fail && not cond then exit !exit_status else ()

let suite name f =
  suite_name := name;
  if print_checkpoints then prerr_endline (name ^ ":");
  f ()
