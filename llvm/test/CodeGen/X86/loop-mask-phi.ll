; RUN: llc -mtriple=x86_64 -mattr=+avx2 -O3 -verify-machineinstrs < %s | FileCheck %s --check-prefix=AVX2
; RUN: llc -mtriple=x86_64 -mattr=+avx2,+avx512f,+avx512vl -O3 -verify-machineinstrs < %s | FileCheck %s --check-prefix=VL
; RUN: llc -mtriple=x86_64 -mattr=+avx2 -O3 -stop-after=codegenprepare < %s | FileCheck %s --check-prefix=CGP

; TODO: Keep the comparison mask wide across the loop PHI on AVX2.

define <8 x float> @advance(<8 x float> %start, <8 x float> %step, <8 x float> %limit) {
; AVX2-LABEL: advance:
; AVX2:       [[LOOP:\.LBB[0-9]+_[0-9]+]]:
; AVX2:       vpmovzxwd
; AVX2:       vpslld
; AVX2:       vblendvps
; AVX2:       vcmpltps
; AVX2:       vextractf128
; AVX2:       vpackssdw
; AVX2:       vtestps
; AVX2:       jne [[LOOP]]
;
; VL-LABEL: advance:
; VL:       [[LOOP:\.LBB[0-9]+_[0-9]+]]:
; VL:       vaddps {{.*}} {%k{{[1-7]}}}
; VL:       vcmpltps {{.*}}, %k{{[1-7]}}
; VL:       jne [[LOOP]]
;
; CGP-LABEL: define <8 x float> @advance(
; CGP:       %active = phi <8 x i1>
entry:
  %initial = fcmp olt <8 x float> %start, %limit
  br label %loop

loop:
  %active = phi <8 x i1> [ %initial, %entry ], [ %next, %loop ]
  %value = phi <8 x float> [ %start, %entry ], [ %updated, %loop ]
  %candidate = fadd <8 x float> %value, %step
  %updated = select <8 x i1> %active, <8 x float> %candidate, <8 x float> %value
  %next = fcmp olt <8 x float> %updated, %limit
  %bits = bitcast <8 x i1> %next to i8
  %done = icmp eq i8 %bits, 0
  br i1 %done, label %exit, label %loop

exit:
  ret <8 x float> %updated
}

define <8 x float> @advance_guarded(<8 x float> %start, <8 x float> %step, <8 x float> %limit) {
; CGP-LABEL: define <8 x float> @advance_guarded(
; CGP:       %active = phi <8 x i1>
entry:
  %initial = fcmp olt <8 x float> %start, %limit
  %initial.bits = bitcast <8 x i1> %initial to i8
  %skip = icmp eq i8 %initial.bits, 0
  br i1 %skip, label %exit, label %preheader

preheader:
  br label %loop

loop:
  %active = phi <8 x i1> [ %initial, %preheader ], [ %next, %loop ]
  %value = phi <8 x float> [ %start, %preheader ], [ %updated, %loop ]
  %candidate = fadd <8 x float> %value, %step
  %updated = select <8 x i1> %active, <8 x float> %candidate, <8 x float> %value
  %next = fcmp olt <8 x float> %updated, %limit
  %bits = bitcast <8 x i1> %next to i8
  %done = icmp eq i8 %bits, 0
  br i1 %done, label %exit, label %loop

exit:
  %result = phi <8 x float> [ %start, %entry ], [ %updated, %loop ]
  ret <8 x float> %result
}

define <8 x i16> @mixed_select_width(<8 x float> %start, <8 x float> %limit, <8 x i16> %yes, <8 x i16> %no, i32 %count) {
; CGP-LABEL: define <8 x i16> @mixed_select_width(
; CGP:       %active = phi <8 x i1>
entry:
  %initial = fcmp olt <8 x float> %start, %limit
  br label %loop

loop:
  %active = phi <8 x i1> [ %initial, %entry ], [ %next, %loop ]
  %iter = phi i32 [ %count, %entry ], [ %remaining, %loop ]
  %result = select <8 x i1> %active, <8 x i16> %yes, <8 x i16> %no
  %as.float = sitofp <8 x i16> %result to <8 x float>
  %next = fcmp olt <8 x float> %as.float, %limit
  %remaining = sub i32 %iter, 1
  %done = icmp eq i32 %remaining, 0
  br i1 %done, label %exit, label %loop

exit:
  ret <8 x i16> %result
}

define <8 x float> @extra_phi_use(<8 x float> %start, <8 x float> %step, <8 x float> %limit, ptr %out) {
; CGP-LABEL: define <8 x float> @extra_phi_use(
; CGP:       %active = phi <8 x i1>
entry:
  %initial = fcmp olt <8 x float> %start, %limit
  br label %loop

loop:
  %active = phi <8 x i1> [ %initial, %entry ], [ %next, %loop ]
  %value = phi <8 x float> [ %start, %entry ], [ %updated, %loop ]
  %candidate = fadd <8 x float> %value, %step
  %updated = select <8 x i1> %active, <8 x float> %candidate, <8 x float> %value
  %active.bits = bitcast <8 x i1> %active to i8
  store volatile i8 %active.bits, ptr %out
  %next = fcmp olt <8 x float> %updated, %limit
  %bits = bitcast <8 x i1> %next to i8
  %done = icmp eq i8 %bits, 0
  br i1 %done, label %exit, label %loop

exit:
  ret <8 x float> %updated
}

define <8 x float> @extra_compare_use(<8 x float> %start, <8 x float> %step, <8 x float> %limit, ptr %out) {
; CGP-LABEL: define <8 x float> @extra_compare_use(
; CGP:       %active = phi <8 x i1>
entry:
  %initial = fcmp olt <8 x float> %start, %limit
  br label %loop

loop:
  %active = phi <8 x i1> [ %initial, %entry ], [ %next, %loop ]
  %value = phi <8 x float> [ %start, %entry ], [ %updated, %loop ]
  %candidate = fadd <8 x float> %value, %step
  %updated = select <8 x i1> %active, <8 x float> %candidate, <8 x float> %value
  %next = fcmp olt <8 x float> %updated, %limit
  %bits = bitcast <8 x i1> %next to i8
  store volatile i8 %bits, ptr %out
  %done = icmp eq i8 %bits, 0
  br i1 %done, label %exit, label %loop

exit:
  ret <8 x float> %updated
}
