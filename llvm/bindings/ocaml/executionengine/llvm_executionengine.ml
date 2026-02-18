(*===-- llvm_executionengine.ml - LLVM OCaml Interface --------*- OCaml -*-===*
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*)

exception Error of string

let () = Callback.register_exception "Llvm_executionengine.Error" (Error "")

external initialize : unit -> bool
  = "llvm_ee_initialize"

type llexecutionengine

type llcompileroptions = {
  opt_level: int;
  code_model: Llvm_target.CodeModel.t;
  no_framepointer_elim: bool;
  enable_fast_isel: bool;
}

let default_compiler_options = {
  opt_level = 0;
  code_model = Llvm_target.CodeModel.JITDefault;
  no_framepointer_elim = false;
  enable_fast_isel = false }

external create : ?options:llcompileroptions -> Llvm.llmodule -> llexecutionengine
  = "llvm_ee_create"
external dispose : llexecutionengine -> unit
  = "llvm_ee_dispose"
external add_module : Llvm.llmodule -> llexecutionengine -> unit
  = "llvm_ee_add_module"
external remove_module : Llvm.llmodule -> llexecutionengine -> unit
  = "llvm_ee_remove_module"
external run_static_ctors : llexecutionengine -> unit
  = "llvm_ee_run_static_ctors"
external run_static_dtors : llexecutionengine -> unit
  = "llvm_ee_run_static_dtors"
external data_layout : llexecutionengine -> Llvm_target.DataLayout.t
  = "llvm_ee_get_data_layout"
external add_global_mapping : Llvm.llvalue -> nativeint -> llexecutionengine -> unit
  = "llvm_ee_add_global_mapping"
external get_global_value_address : string -> llexecutionengine -> nativeint
  = "llvm_ee_get_global_value_address"
external get_function_address : string -> llexecutionengine -> nativeint
  = "llvm_ee_get_function_address"

(* The following are not bound. Patches are welcome.
target_machine : llexecutionengine -> Llvm_target.TargetMachine.t
 *)
