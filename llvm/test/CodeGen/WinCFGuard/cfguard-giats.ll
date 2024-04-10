; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-w64-windows-gnu | FileCheck %s
; Control Flow Guard is currently only available on Windows

declare dllimport i32 @target_func1()
declare dllimport i32 @target_func2()
declare dllimport i32 @target_func3()
@ptrs = dso_local local_unnamed_addr global [2 x ptr] [ptr @target_func2, ptr @target_func3], align 16

; Test address-taken functions from imported DLLs are correctly added to the 
; Guard Address-Taken IAT Entry (.giats) and Guard Function ID (.gfids) sections.
define i32 @func_cf_giats1() {
entry:
  ; Since it is a dllimport, target_func1 will be represented as "__imp_target_func1" when it is
  ; stored in the function pointer. Therefore, the .giats section must contain "__imp_target_func1".
  ; Unlike MSVC, we also have "target_func1" in the .gfids section, since this is not a security risk.
  %func_ptr = alloca ptr, align 8
  store ptr @target_func1, ptr %func_ptr, align 8
  %0 = load ptr, ptr %func_ptr, align 8
  %1 = call i32 %0()
  ; target_func2 is called directly from a global array, so should only appear in the .gfids section.
  %2 = load ptr, ptr @ptrs, align 8
  %3 = call i32 %2()
  ; target_func3 is called both via a stored function pointer (as with target_func1) and via a gloabl
  ; array (as with target_func2), so "target_func3" must appear in .gfids and "__imp_target_func3" in .giats.
  store ptr @target_func3, ptr %func_ptr, align 8
  %4 = load ptr, ptr %func_ptr, align 8
  %5 = call i32 %4()
  %6 = load ptr, ptr getelementptr inbounds ([2 x ptr], ptr @ptrs, i64 0, i64 1), align 8
  %7 = call i32 %6()
  ret i32 %5
}

; CHECK-LABEL: .section .gfids$y,"dr"
; CHECK-NEXT:  .symidx target_func1
; CHECK-NEXT:  .symidx target_func2
; CHECK-NEXT:  .symidx target_func3
; CHECK-NOT:   .symidx
; CHECK-LABEL: .section .giats$y,"dr"
; CHECK-NEXT:  .symidx __imp_target_func1
; CHECK-NEXT:  .symidx __imp_target_func3
; CHECK-NOT:   .symidx

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
