; RUN: llc -mtriple aarch64-arm-none-eabi -enable-machine-outliner \
; RUN:  -verify-machineinstrs %s -o - | FileCheck %s

@v = common dso_local global ptr null, align 8

; CHECK-LABEL:  foo:                                    // @foo
; CHECK-NEXT:   // %bb.0:                               // %entry
; CHECK-NEXT:       paciasp
; CHECK-NOT:        OUTLINED_FUNCTION_
; CHECK:            retaa
define dso_local void @foo(i32 %x) #0 {
entry:
  %0 = zext i32 %x to i64
  %vla = alloca i32, i64 %0, align 4
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  ret void
}

; CHECK-LABEL:  bar:                                    // @bar
; CHECK-NEXT:   // %bb.0:                               // %entry
; CHECK-NEXT:       paciasp
; CHECK-NOT:        OUTLINED_FUNCTION_
; CHECK:            retaa
define dso_local void @bar(i32 %x) #0 {
entry:
  %0 = zext i32 %x to i64
  %vla = alloca i32, i64 %0, align 4
  store volatile ptr null, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  store volatile ptr %vla, ptr @v, align 8
  ret void
}

attributes #0 = { nounwind "target-features"="+v8.3a" "frame-pointer"="all" "sign-return-address"="all" "sign-return-address-key"="a_key" }

; CHECK-NOT:  OUTLINED_FUNCTION_
