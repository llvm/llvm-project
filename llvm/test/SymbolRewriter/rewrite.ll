; RUN: opt -mtriple i686-win32 -passes='rewrite-symbols' -rewrite-map-file %p/rewrite.map \
; RUN:   %s -o - | llvm-dis | FileCheck %s

declare void @source_function()
@source_variable = external global i32
declare void @source_function_pattern_function()
declare void @source_function_pattern_multiple_function_matches()
@source_variable_pattern_variable = external global i32
@source_variable_pattern_multiple_variable_matches = external global i32
declare void @"\01naked_source_function"()
declare void @"\01__imp_missing_global_leader_prefix"()

declare i32 @first_callee()
declare i32 @second_callee()
define i32 @caller() {
  %rhs = call i32 @first_callee()
  %lhs = call i32 @second_callee()
  %res = add i32 %rhs, %lhs
  ret i32 %res
}

%struct.S = type { i8 }
@_ZN1SC1Ev = alias void (%struct.S*), void (%struct.S*)* @_ZN1SC2Ev
define void @_ZN1SC2Ev(%struct.S* %this) unnamed_addr align 2 {
entry:
  %this.addr = alloca %struct.S*, align 4
  store %struct.S* %this, %struct.S** %this.addr, align 4
  ret void
}

$source_comdat_function = comdat any
define dllexport void @source_comdat_function() comdat($source_comdat_function) {
entry:
  ret void
}

$source_comdat_function_1 = comdat exactmatch
define dllexport void @source_comdat_function_1() comdat($source_comdat_function_1) {
entry:
  ret void
}

$source_comdat_variable = comdat largest
@source_comdat_variable = global i32 32, comdat($source_comdat_variable)

$source_comdat_variable_1 = comdat nodeduplicate
@source_comdat_variable_1 = global i32 64, comdat($source_comdat_variable_1)

; CHECK: $target_comdat_function = comdat any
; CHECK: $target_comdat_function_1 = comdat exactmatch
; CHECK: $target_comdat_variable = comdat largest
; CHECK: $target_comdat_variable_1 = comdat nodeduplicate

; CHECK: @target_variable = external global i32
; CHECK-NOT: @source_variable = external global i32
; CHECK: @target_pattern_variable = external global i32
; CHECK-NOT: @source_pattern_variable = external global i32
; CHECK: @target_pattern_multiple_variable_matches = external global i32
; CHECK-NOT: @source_pattern_multiple_variable_matches = external global i32
; CHECK: @target_comdat_variable = global i32 32, comdat
; CHECK-NOT: @source_comdat_variable = global i32 32, comdat
; CHECK: @target_comdat_variable_1 = global i32 64, comdat
; CHECK-NOT: @source_comdat_variable_1 = global i32 64, comdat

; CHECK: declare void @target_function()
; CHECK-NOT: declare void @source_function()
; CHECK: declare void @target_pattern_function()
; CHECK-NOT: declare void @source_function_pattern_function()
; CHECK: declare void @target_pattern_multiple_function_matches()
; CHECK-NOT: declare void @source_function_pattern_multiple_function_matches()
; CHECK: declare void @naked_target_function()
; CHECK-NOT: declare void @"\01naked_source_function"()
; CHECK-NOT: declare void @"\01__imp__imported_function"()
; CHECK: declare void @"\01__imp_missing_global_leader_prefix"()
; CHECK-NOT: declare void @"\01__imp_DO_NOT_REWRITE"()

; CHECK: declare i32 @renamed_callee()
; CHECK-NOT: declare i32 @first_callee()
; CHECK: declare i32 @second_callee()
; CHECK: define i32 @caller() {
; CHECK:   %rhs = call i32 @renamed_callee()
; CHECK-NOT: %rhs = call i32 @first_callee()
; CHECK:   %lhs = call i32 @second_callee()
; CHECK:   %res = add i32 %rhs, %lhs
; CHECK:   ret i32 %res
; CHECK: }

; CHECK: define dllexport void @target_comdat_function() comdat
; CHECK-NOT: define dllexport void @source_comdat_function() comdat
; CHECK: define dllexport void @target_comdat_function_1() comdat
; CHECK-NOT: define dllexport void @source_comdat_function_1() comdat

