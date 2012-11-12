; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+sse41,-avx < %s | FileCheck %s --check-prefix SSE41
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+avx,-avx2 < %s | FileCheck %s --check-prefix AVX

define i32 @veccond128(<4 x i32> %input) {
entry:
  %0 = bitcast <4 x i32> %input to i128
  %1 = icmp ne i128 %0, 0
  br i1 %1, label %if-true-block, label %endif-block

if-true-block:                                    ; preds = %entry
  ret i32 0
endif-block:                                      ; preds = %entry,
  ret i32 1
; SSE41: veccond128
; SSE41: ptest
; SSE41: ret
; AVX:   veccond128
; AVX:   vptest %xmm{{.*}}, %xmm{{.*}}
; AVX:   ret
}

define i32 @veccond256(<8 x i32> %input) {
entry:
  %0 = bitcast <8 x i32> %input to i256
  %1 = icmp ne i256 %0, 0
  br i1 %1, label %if-true-block, label %endif-block

if-true-block:                                    ; preds = %entry
  ret i32 0
endif-block:                                      ; preds = %entry,
  ret i32 1
; SSE41: veccond256
; SSE41: por
; SSE41: ptest
; SSE41: ret
; AVX:   veccond256
; AVX:   vptest %ymm{{.*}}, %ymm{{.*}}
; AVX:   ret
}

define i32 @veccond512(<16 x i32> %input) {
entry:
  %0 = bitcast <16 x i32> %input to i512
  %1 = icmp ne i512 %0, 0
  br i1 %1, label %if-true-block, label %endif-block

if-true-block:                                    ; preds = %entry
  ret i32 0
endif-block:                                      ; preds = %entry,
  ret i32 1
; SSE41: veccond512
; SSE41: por
; SSE41: por
; SSE41: por
; SSE41: ptest
; SSE41: ret
; AVX:   veccond512
; AVX:   vorps
; AVX:   vptest %ymm{{.*}}, %ymm{{.*}}
; AVX:   ret
}

define i32 @vectest128(<4 x i32> %input) {
entry:
  %0 = bitcast <4 x i32> %input to i128
  %1 = icmp ne i128 %0, 0
  %2 = zext i1 %1 to i32
  ret i32 %2
; SSE41: vectest128
; SSE41: ptest
; SSE41: ret
; AVX:   vectest128
; AVX:   vptest %xmm{{.*}}, %xmm{{.*}}
; AVX:   ret
}

define i32 @vectest256(<8 x i32> %input) {
entry:
  %0 = bitcast <8 x i32> %input to i256
  %1 = icmp ne i256 %0, 0
  %2 = zext i1 %1 to i32
  ret i32 %2
; SSE41: vectest256
; SSE41: por
; SSE41: ptest
; SSE41: ret
; AVX:   vectest256
; AVX:   vptest %ymm{{.*}}, %ymm{{.*}}
; AVX:   ret
}

define i32 @vectest512(<16 x i32> %input) {
entry:
  %0 = bitcast <16 x i32> %input to i512
  %1 = icmp ne i512 %0, 0
  %2 = zext i1 %1 to i32
  ret i32 %2
; SSE41: vectest512
; SSE41: por
; SSE41: por
; SSE41: por
; SSE41: ptest
; SSE41: ret
; AVX:   vectest512
; AVX:   vorps
; AVX:   vptest %ymm{{.*}}, %ymm{{.*}}
; AVX:   ret
}

define i32 @vecsel128(<4 x i32> %input, i32 %a, i32 %b) {
entry:
  %0 = bitcast <4 x i32> %input to i128
  %1 = icmp ne i128 %0, 0
  %2 = select i1 %1, i32 %a, i32 %b
  ret i32 %2
; SSE41: vecsel128
; SSE41: ptest
; SSE41: ret
; AVX:   vecsel128
; AVX:   vptest %xmm{{.*}}, %xmm{{.*}}
; AVX:   ret
}

define i32 @vecsel256(<8 x i32> %input, i32 %a, i32 %b) {
entry:
  %0 = bitcast <8 x i32> %input to i256
  %1 = icmp ne i256 %0, 0
  %2 = select i1 %1, i32 %a, i32 %b
  ret i32 %2
; SSE41: vecsel256
; SSE41: por
; SSE41: ptest
; SSE41: ret
; AVX:   vecsel256
; AVX:   vptest %ymm{{.*}}, %ymm{{.*}}
; AVX:   ret
}

define i32 @vecsel512(<16 x i32> %input, i32 %a, i32 %b) {
entry:
  %0 = bitcast <16 x i32> %input to i512
  %1 = icmp ne i512 %0, 0
  %2 = select i1 %1, i32 %a, i32 %b
  ret i32 %2
; SSE41: vecsel512
; SSE41: por
; SSE41: por
; SSE41: por
; SSE41: ptest
; SSE41: ret
; AVX:   vecsel512
; AVX:   vorps
; AVX:   vptest %ymm{{.*}}, %ymm{{.*}}
; AVX:   ret
}
