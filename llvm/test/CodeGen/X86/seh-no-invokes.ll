; RUN: llc < %s | FileCheck %s

; Generated with this C source:
; static __forceinline void __cpuid() { __asm__(""); }
; void f() {
;   __try {
;     __cpuid();
;   } __except (1) {
;   }
; }

; When running clang at -O1, we can end up deleting unreachable SEH catchpads
; without running GlobalDCE to remove the associated filter. This used to
; result in references to undefined labels. Now we check that we emit the
; label. This was PR30431.

; CHECK-LABEL: _f:                                     # @f
; CHECK: .set Lf$parent_frame_offset, 0
; CHECK: retl

; CHECK-LABEL: "?filt$0@0@f@@":                        # @"\01?filt$0@0@f@@"
; CHECK: movl    $Lf$parent_frame_offset,

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.0.24210"

define void @f() #0 personality ptr @_except_handler3 {
__try.cont:
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(ptr nonnull %__exception_code)
  call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"() #3, !srcloc !1
  ret void
}

; Function Attrs: nounwind
define internal i32 @"\01?filt$0@0@f@@"() #1 {
entry:
  %0 = tail call ptr @llvm.frameaddress(i32 1)
  %1 = tail call ptr @llvm.eh.recoverfp(ptr @f, ptr %0)
  %2 = tail call ptr @llvm.localrecover(ptr @f, ptr %1, i32 0)
  %3 = getelementptr inbounds i8, ptr %0, i32 -20
  %4 = load ptr, ptr %3, align 4
  %5 = getelementptr inbounds { ptr, ptr }, ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 4
  %7 = load i32, ptr %6, align 4
  store i32 %7, ptr %2, align 4
  ret i32 1
}

; Function Attrs: nounwind readnone
declare ptr @llvm.frameaddress(i32) #2

; Function Attrs: nounwind readnone
declare ptr @llvm.eh.recoverfp(ptr, ptr) #2

; Function Attrs: nounwind readnone
declare ptr @llvm.localrecover(ptr, ptr, i32) #2

declare i32 @_except_handler3(...)

; Function Attrs: nounwind
declare void @llvm.localescape(...) #3

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (trunk 282900) (llvm/trunk 282903)"}
!1 = !{i32 48}
