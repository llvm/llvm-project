; RUN: llc < %s | FileCheck %s

; Based on this code:
;
; extern "C" int array[4];
; extern "C" void global_array(int idx1, int idx2, int idx3) {
;   try {
;     array[idx1] = 111;
;     throw;
;   } catch (...) {
;     array[idx2] = 222;
;   }
;   array[idx3] = 333;
; }
; extern "C" __declspec(dllimport) int imported;
; extern "C" void access_imported() {
;   try {
;     imported = 111;
;     throw;
;   } catch (...) {
;     imported = 222;
;   }
;   imported = 333;
; }

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"

%eh.ThrowInfo = type { i32, i32, i32, i32 }

@array = external global [4 x i32], align 16
@imported = external dllimport global i32, align 4

; Function Attrs: uwtable
define void @global_array(i32 %idx1, i32 %idx2, i32 %idx3) #0 personality ptr @__CxxFrameHandler3 {
entry:
  %idxprom = sext i32 %idx1 to i64
  %arrayidx = getelementptr inbounds [4 x i32], ptr @array, i64 0, i64 %idxprom
  store i32 111, ptr %arrayidx, align 4, !tbaa !2
  invoke void @_CxxThrowException(ptr null, ptr null) #1
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [ptr null, i32 64, ptr null]
  %idxprom1 = sext i32 %idx2 to i64
  %arrayidx2 = getelementptr inbounds [4 x i32], ptr @array, i64 0, i64 %idxprom1
  store i32 222, ptr %arrayidx2, align 4, !tbaa !2
  catchret from %0 to label %try.cont

try.cont:                                         ; preds = %catch
  %idxprom3 = sext i32 %idx3 to i64
  %arrayidx4 = getelementptr inbounds [4 x i32], ptr @array, i64 0, i64 %idxprom3
  store i32 333, ptr %arrayidx4, align 4, !tbaa !2
  ret void

unreachable:                                      ; preds = %entry
  unreachable
}

; CHECK-LABEL: global_array: # @global_array
; CHECK: pushq %rbp
; 	First array access
; CHECK: movslq  %ecx, %[[idx:[^ ]*]]
; CHECK: leaq    array(%rip), %[[base:[^ ]*]]
; CHECK: movl    $111, (%[[base]],%[[idx]],4)
;	Might throw an exception and return to below...
; CHECK: callq   _CxxThrowException
; 	Third array access must remat the address of array
; CHECK: movslq  {{.*}}, %[[idx:[^ ]*]]
; CHECK: leaq    array(%rip), %[[base:[^ ]*]]
; CHECK: movl    $333, (%[[base]],%[[idx]],4)
; CHECK: popq %rbp
; CHECK: retq

; CHECK: "?catch${{[0-9]+}}@?0?global_array@4HA":
; CHECK: pushq   %rbp
; CHECK: movslq  {{.*}}, %[[idx:[^ ]*]]
; CHECK: leaq    array(%rip), %[[base:[^ ]*]]
; CHECK: movl    $222, (%[[base]],%[[idx]],4)
; CHECK: popq    %rbp
; CHECK: retq                            # CATCHRET

declare void @_CxxThrowException(ptr, ptr)

declare i32 @__CxxFrameHandler3(...)

; Function Attrs: uwtable
define void @access_imported() #0 personality ptr @__CxxFrameHandler3 {
entry:
  store i32 111, ptr @imported, align 4, !tbaa !2
  invoke void @_CxxThrowException(ptr null, ptr null) #1
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [ptr null, i32 64, ptr null]
  store i32 222, ptr @imported, align 4, !tbaa !2
  catchret from %0 to label %try.cont

try.cont:                                         ; preds = %catch
  store i32 333, ptr @imported, align 4, !tbaa !2
  ret void

unreachable:                                      ; preds = %entry
  unreachable
}

; CHECK-LABEL: access_imported: # @access_imported
; CHECK: pushq %rbp
; CHECK: movq    __imp_imported(%rip), %[[base:[^ ]*]]
; CHECK: movl    $111, (%[[base]])
;	Might throw an exception and return to below...
; CHECK: callq   _CxxThrowException
; 	Third access must reload the address of imported
; CHECK: movq    __imp_imported(%rip), %[[base:[^ ]*]]
; CHECK: movl    $333, (%[[base]])
; CHECK: popq %rbp
; CHECK: retq

; CHECK: "?catch${{[0-9]+}}@?0?access_imported@4HA":
; CHECK: pushq   %rbp
; CHECK: movq    __imp_imported(%rip), %[[base:[^ ]*]]
; CHECK: movl    $222, (%[[base]])
; CHECK: popq    %rbp
; CHECK: retq                            # CATCHRET


attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.8.0 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
