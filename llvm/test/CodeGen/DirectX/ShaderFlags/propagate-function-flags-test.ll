; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.7-library"

; CHECK: ; Combined Shader Flags for Module
; CHECK-NEXT: ; Shader Flags Value: 0x00000044
; CHECK-NEXT: ;
; CHECK-NEXT: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Double-precision floating point
; CHECK-NEXT: ;       Double-precision extensions for 11.1
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: ;
; CHECK-NEXT: ; Shader Flags for Module Functions

; Call Graph of test source
; main        -> [get_fptoui_flag, get_sitofp_fdiv_flag]
; get_fptoui_flag  -> [get_sitofp_uitofp_flag, call_get_uitofp_flag]
; get_sitofp_uitofp_flag  -> [call_get_fptoui_flag, call_get_sitofp_flag]
; call_get_fptoui_flag  -> [get_fptoui_flag]
; get_sitofp_fdiv_flag -> [get_no_flags, get_all_doubles_flags]
; get_all_doubles_flags -> [call_get_sitofp_fdiv_flag]
; call_get_sitofp_fdiv_flag  -> [get_sitofp_fdiv_flag]
; call_get_sitofp_flag  -> [get_sitofp_flag]
; call_get_uitofp_flag  -> [get_uitofp_flag]
; get_sitofp_flag  -> []
; get_uitofp_flag  -> []
; get_no_flags  -> []
;
; Strongly Connected Component in the CG
;  [get_fptoui_flag, get_sitofp_uitofp_flag, call_get_fptoui_flag]
;  [get_sitofp_fdiv_flag, get_all_doubles_flags, call_get_sitofp_fdiv_flag]

;
; CHECK: ; Function get_sitofp_flag  : 0x00000044
define double @get_sitofp_flag(i32 noundef %0) local_unnamed_addr #0 {
  %2 = sitofp i32 %0 to double
  ret double %2
}

; CHECK: ; Function call_get_sitofp_flag : 0x00000044
define double @call_get_sitofp_flag(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call double @get_sitofp_flag(i32 noundef %0)
  ret double %2
}

; CHECK: ; Function get_uitofp_flag : 0x00000044
define double @get_uitofp_flag(i32 noundef %0) local_unnamed_addr #0 {
  %2 = uitofp i32 %0 to double
  ret double %2
}

; CHECK: ; Function call_get_uitofp_flag : 0x00000044
define double @call_get_uitofp_flag(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call double @get_uitofp_flag(i32 noundef %0)
  ret double %2
}

; CHECK: ; Function call_get_fptoui_flag : 0x00000044
define double @call_get_fptoui_flag(double noundef %0) local_unnamed_addr #0 {
  %2 = tail call double @get_fptoui_flag(double noundef %0)
  ret double %2
}

; CHECK: ; Function get_fptoui_flag : 0x00000044
define double @get_fptoui_flag(double noundef %0) local_unnamed_addr #0 {
  %2 = fcmp ugt double %0, 5.000000e+00
  br i1 %2, label %6, label %3

3:                                                ; preds = %1
  %4 = fptoui double %0 to i64
  %5 = tail call double @get_sitofp_uitofp_flag(i64 noundef %4)
  br label %9

6:                                                ; preds = %1
  %7 = fptoui double %0 to i32
  %8 = tail call double @call_get_uitofp_flag(i32 noundef %7)
  br label %9

9:                                                ; preds = %6, %3
  %10 = phi double [ %5, %3 ], [ %8, %6 ]
  ret double %10
}

; CHECK: ; Function get_sitofp_uitofp_flag : 0x00000044
define double @get_sitofp_uitofp_flag(i64 noundef %0) local_unnamed_addr #0 {
  %2 = icmp ult i64 %0, 6
  br i1 %2, label %3, label %7

3:                                                ; preds = %1
  %4 = add nuw nsw i64 %0, 1
  %5 = uitofp i64 %4 to double
  %6 = tail call double @call_get_fptoui_flag(double noundef %5)
  br label %10

7:                                                ; preds = %1
  %8 = trunc i64 %0 to i32
  %9 = tail call double @call_get_sitofp_flag(i32 noundef %8)
  br label %10

10:                                               ; preds = %7, %3
  %11 = phi double [ %6, %3 ], [ %9, %7 ]
  ret double %11
}

; CHECK: ; Function get_no_flags : 0x00000000
define i32 @get_no_flags(i32 noundef %0) local_unnamed_addr #0 {
  %2 = mul nsw i32 %0, %0
  ret i32 %2
}

; CHECK: ; Function call_get_sitofp_fdiv_flag : 0x00000044
define i32 @call_get_sitofp_fdiv_flag(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i32 %0, 0
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = mul nsw i32 %0, %0
  br label %7

5:                                                ; preds = %1
  %6 = tail call double @get_sitofp_fdiv_flag(i32 noundef 0)
  br label %7

7:                                                ; preds = %5, %3
  %8 = phi i32 [ %4, %3 ], [ 0, %5 ]
  ret i32 %8
}

; CHECK: ; Function get_sitofp_fdiv_flag : 0x00000044
define double @get_sitofp_fdiv_flag(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 5
  br i1 %2, label %3, label %6

3:                                                ; preds = %1
  %4 = tail call i32 @get_no_flags(i32 noundef %0)
  %5 = sitofp i32 %4 to double
  br label %9

6:                                                ; preds = %1
  %7 = tail call double @get_all_doubles_flags(i32 noundef %0)
  %8 = fdiv double %7, 3.000000e+00
  br label %9

9:                                                ; preds = %6, %3
  %10 = phi double [ %5, %3 ], [ %8, %6 ]
  ret double %10
}

; CHECK: ; Function get_all_doubles_flags : 0x00000044
define double @get_all_doubles_flags(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call i32 @call_get_sitofp_fdiv_flag(i32 noundef %0)
  %3 = icmp eq i32 %2, 0
  %4 = select i1 %3, double 1.000000e+01, double 1.000000e+02
  ret double %4
}

; CHECK: ; Function main : 0x00000044
define i32 @main() local_unnamed_addr #0 {
  %1 = tail call double @get_fptoui_flag(double noundef 1.000000e+00)
  %2 = tail call double @get_sitofp_fdiv_flag(i32 noundef 4)
  %3 = fadd double %1, %2
  %4 = fcmp ogt double %3, 0.000000e+00
  %5 = zext i1 %4 to i32
  ret i32 %5
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
