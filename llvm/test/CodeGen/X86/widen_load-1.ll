; RUN: llc -stack-symbol-ordering=0 %s -o - -mattr=-avx -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=SSE
; RUN: llc -stack-symbol-ordering=0 %s -o - -mattr=+avx -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=AVX
; PR4891
; PR5626

; This load should be before the call, not after.

; SSE: movsd     compl+128(%rip), %xmm0
; SSE: movaps  %xmm0, (%rsp)
; SSE: callq   killcommon

; AVX: vmovsd     compl+128(%rip), %xmm0
; AVX: vmovaps  %xmm0, (%rsp)
; AVX: callq   killcommon

@compl = linkonce dso_local global [20 x i64] zeroinitializer, align 64 ; <ptr> [#uses=1]

declare void @killcommon(ptr noalias)

define dso_local void @reset(ptr noalias %garbage1) {
"file complex.c, line 27, bb1":
  %changed = alloca i32, align 4                  ; <ptr> [#uses=3]
  br label %"file complex.c, line 27, bb13"

"file complex.c, line 27, bb13":                  ; preds = %"file complex.c, line 27, bb1"
  store i32 0, ptr %changed, align 4
  %r2 = getelementptr float, ptr @compl, i64 32 ; <ptr> [#uses=1]
  %r4 = load <2 x float>, ptr %r2, align 4            ; <<2 x float>> [#uses=1]
  call void @killcommon(ptr %changed)
  br label %"file complex.c, line 34, bb4"

"file complex.c, line 34, bb4":                   ; preds = %"file complex.c, line 27, bb13"
  %r5 = load i32, ptr %changed, align 4               ; <i32> [#uses=1]
  %r6 = icmp eq i32 %r5, 0                        ; <i1> [#uses=1]
  %r7 = zext i1 %r6 to i32                        ; <i32> [#uses=1]
  %r8 = icmp ne i32 %r7, 0                        ; <i1> [#uses=1]
  br i1 %r8, label %"file complex.c, line 34, bb7", label %"file complex.c, line 27, bb5"

"file complex.c, line 27, bb5":                   ; preds = %"file complex.c, line 34, bb4"
  br label %"file complex.c, line 35, bb6"

"file complex.c, line 35, bb6":                   ; preds = %"file complex.c, line 27, bb5"
  %r11 = ptrtoint ptr %garbage1 to i64   ; <i64> [#uses=1]
  %r12 = inttoptr i64 %r11 to ptr        ; <ptr> [#uses=1]
  store <2 x float> %r4, ptr %r12, align 4
  br label %"file complex.c, line 34, bb7"

"file complex.c, line 34, bb7":                   ; preds = %"file complex.c, line 35, bb6", %"file complex.c, line 34, bb4"
  ret void
}
