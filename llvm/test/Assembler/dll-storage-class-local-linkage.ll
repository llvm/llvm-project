;; Check that an error is emitted for local linkage symbols with DLL storage class.

; RUN: rm -rf %t && split-file %s %t

; RUN: not llvm-as %t/internal_function_dllexport.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llvm-as %t/internal_variable_dllexport.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llvm-as %t/internal_alias_dllexport.ll -o /dev/null 2>&1 | FileCheck %s

; RUN: not llvm-as %t/private_function_dllexport.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llvm-as %t/private_variable_dllexport.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llvm-as %t/private_alias_dllexport.ll -o /dev/null 2>&1 | FileCheck %s

; RUN: not llvm-as %t/internal_function_dllimport.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llvm-as %t/internal_variable_dllimport.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llvm-as %t/internal_alias_dllimport.ll -o /dev/null 2>&1 | FileCheck %s

; RUN: not llvm-as %t/private_function_dllimport.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llvm-as %t/private_variable_dllimport.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llvm-as %t/private_alias_dllimport.ll -o /dev/null 2>&1 | FileCheck %s


; CHECK: symbol with local linkage cannot have a DLL storage class


;--- internal_function_dllexport.ll

define internal dllexport void @function() {
entry:
  ret void
}

;--- internal_variable_dllexport.ll

@var = internal dllexport global i32 0

;--- internal_alias_dllexport.ll

@global = global i32 0
@alias = internal dllexport alias i32, ptr @global

;--- private_function_dllexport.ll

define private dllexport void @function() {
entry:
  ret void
}

;--- private_variable_dllexport.ll

@var = private dllexport global i32 0

;--- private_alias_dllexport.ll

@global = global i32 0
@alias = private dllexport alias i32, ptr @global


;--- internal_function_dllimport.ll

define internal dllimport void @function() {
entry:
  ret void
}

;--- internal_variable_dllimport.ll

@var = internal dllimport global i32 0

;--- internal_alias_dllimport.ll

@global = global i32 0
@alias = internal dllimport alias i32, ptr @global

;--- private_function_dllimport.ll

define private dllimport void @function() {
entry:
  ret void
}

;--- private_variable_dllimport.ll

@var = private dllimport global i32 0

;--- private_alias_dllimport.ll

@global = global i32 0
@alias = private dllimport alias i32, ptr @global
