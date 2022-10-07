; RUN: opt < %s -sink -S | FileCheck %s
; RUN: opt < %s -aa-pipeline='basic-aa' -passes='sink' -S | FileCheck %s

@A = external global i32
@B = external global i32

; Sink should sink the load past the store (which doesn't overlap) into
; the block that uses it.

;      CHECK-LABEL: @foo(
;      CHECK: true:
; CHECK-NEXT: %l = load i32, i32* @A
; CHECK-NEXT: ret i32 %l

define i32 @foo(i1 %z) {
  %l = load i32, i32* @A
  store i32 0, i32* @B
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

; But don't sink load volatiles...

;      CHECK-LABEL: @foo2(
;      CHECK: load volatile
; CHECK-NEXT: store i32

define i32 @foo2(i1 %z) {
  %l = load volatile i32, i32* @A
  store i32 0, i32* @B
  br i1 %z, label %true, label %false
true:
  ret i32 %l
false:
  ret i32 0
}

; Sink to the nearest post-dominator

;      CHECK-LABEL: @diamond(
;      CHECK: X:
; CHECK-NEXT: phi
; CHECK-NEXT: mul nsw
; CHECK-NEXT: sub

define i32 @diamond(i32 %a, i32 %b, i32 %c) {
  %1 = mul nsw i32 %c, %b
  %2 = icmp sgt i32 %a, 0
  br i1 %2, label %B0, label %B1

B0:                                       ; preds = %0
  br label %X

B1:                                      ; preds = %0
  br label %X

X:                                     ; preds = %5, %3
  %.01 = phi i32 [ %c, %B0 ], [ %a, %B1 ]
  %R = sub i32 %1, %.01
  ret i32 %R
}

; We shouldn't sink constant sized allocas from the entry block, since CodeGen
; interprets allocas outside the entry block as dynamically sized stack objects.

; CHECK-LABEL: @alloca_nosink
; CHECK: entry:
; CHECK-NEXT: alloca
define i32 @alloca_nosink(i32 %a, i32 %b) {
entry:
  %0 = alloca i32
  %1 = icmp ne i32 %a, 0
  br i1 %1, label %if, label %endif

if:
  %2 = getelementptr i32, i32* %0, i32 1
  store i32 0, i32* %0
  store i32 1, i32* %2
  %3 = getelementptr i32, i32* %0, i32 %b
  %4 = load i32, i32* %3
  ret i32 %4

endif:
  ret i32 0
}

; Make sure we sink dynamic sized allocas

; CHECK-LABEL: @alloca_sink_dynamic
; CHECK: entry:
; CHECK-NOT: alloca
; CHECK: if:
; CHECK-NEXT: alloca
define i32 @alloca_sink_dynamic(i32 %a, i32 %b, i32 %size) {
entry:
  %0 = alloca i32, i32 %size
  %1 = icmp ne i32 %a, 0
  br i1 %1, label %if, label %endif

if:
  %2 = getelementptr i32, i32* %0, i32 1
  store i32 0, i32* %0
  store i32 1, i32* %2
  %3 = getelementptr i32, i32* %0, i32 %b
  %4 = load i32, i32* %3
  ret i32 %4

endif:
  ret i32 0
}

; We also want to sink allocas that are not in the entry block.  These
; will already be considered as dynamically sized stack objects, so sinking
; them does no further damage.

; CHECK-LABEL: @alloca_sink_nonentry
; CHECK: if0:
; CHECK-NOT: alloca
; CHECK: if:
; CHECK-NEXT: alloca
define i32 @alloca_sink_nonentry(i32 %a, i32 %b, i32 %c) {
entry:
  %cmp = icmp ne i32 %c, 0
  br i1 %cmp, label %endif, label %if0

if0:
  %0 = alloca i32
  %1 = icmp ne i32 %a, 0
  br i1 %1, label %if, label %endif

if:
  %2 = getelementptr i32, i32* %0, i32 1
  store i32 0, i32* %0
  store i32 1, i32* %2
  %3 = getelementptr i32, i32* %0, i32 %b
  %4 = load i32, i32* %3
  ret i32 %4

endif:
  ret i32 0
}
