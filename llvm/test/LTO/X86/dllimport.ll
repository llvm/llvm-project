; Test requiring LTO to remove the __imp_ prefix for locally imported COFF
; symbols (mirroring how lld handles these symbols).
; RUN: llvm-as %s -o %t.obj
; RUN: llvm-as %S/Inputs/dllimport.ll -o %t2.obj
; RUN: llvm-lto2 run -r=%t.obj,main,px -r %t.obj,__imp_?foo@@YAHXZ -r %t2.obj,?foo@@YAHXZ,p -o %t3 %t.obj %t2.obj -save-temps
; RUN: llvm-dis %t3.0.0.preopt.bc -o - | FileCheck %s

; CHECK: define dso_local i32 @"?foo@@YAHXZ"()

; ModuleID = 'a.obj'
source_filename = "a.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"

; Function Attrs: norecurse nounwind sspstrong uwtable
define dso_local i32 @main() local_unnamed_addr {
entry:
  %call = tail call i32 @"?foo@@YAHXZ"()
  ret i32 %call
}

declare dllimport i32 @"?foo@@YAHXZ"() local_unnamed_addr

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ThinLTO", i32 0}

^0 = module: (path: "a.obj", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "?foo@@YAHXZ") ; guid = 2709792123250749187
^2 = gv: (name: "main", summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 1, live: 0, dsoLocal: 1), insts: 2, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 1, returnDoesNotAlias: 0), calls: ((callee: ^1))))) ; guid = 15822663052811949562
