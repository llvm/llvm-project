; RUN: opt -passes=sandbox-vectorizer -sbvec-vec-reg-bits=1024 -sbvec-allow-non-pow2 -sbvec-passes="seed-collection<tr-save,bottom-up-vec,tr-accept>" -sbvec-allow-files="some_other_file" %s -S | FileCheck %s --check-prefix=ALLOW_OTHER
; RUN: opt -passes=sandbox-vectorizer -sbvec-vec-reg-bits=1024 -sbvec-allow-non-pow2 -sbvec-passes="seed-collection<tr-save,bottom-up-vec,tr-accept>" -sbvec-allow-files="allow_files.ll" %s -S | FileCheck %s --check-prefix=ALLOW_THIS
; RUN: opt -passes=sandbox-vectorizer -sbvec-vec-reg-bits=1024 -sbvec-allow-non-pow2 -sbvec-passes="seed-collection<tr-save,bottom-up-vec,tr-accept>" -sbvec-allow-files="al.*_files.ll" %s -S | FileCheck %s --check-prefix=ALLOW_REGEX
; RUN: opt -passes=sandbox-vectorizer -sbvec-vec-reg-bits=1024 -sbvec-allow-non-pow2 -sbvec-passes="seed-collection<tr-save,bottom-up-vec,tr-accept>" -sbvec-allow-files="some_file,.*_files.ll,some_other_file" %s -S | FileCheck %s --check-prefix=ALLOW_REGEX_CSV
; RUN: opt -passes=sandbox-vectorizer -sbvec-vec-reg-bits=1024 -sbvec-allow-non-pow2 -sbvec-passes="seed-collection<tr-save,bottom-up-vec,tr-accept>" -sbvec-allow-files="allow" %s -S | FileCheck %s --check-prefix=ALLOW_BAD_REGEX
; RUN: opt -passes=sandbox-vectorizer -sbvec-vec-reg-bits=1024 -sbvec-allow-non-pow2 -sbvec-passes="seed-collection<tr-save,bottom-up-vec,tr-accept>" -sbvec-allow-files="some_file,some_other_file1,some_other_file2" %s -S | FileCheck %s --check-prefix=ALLOW_OTHER_CSV
; RUN: opt -passes=sandbox-vectorizer -sbvec-vec-reg-bits=1024 -sbvec-allow-non-pow2 -sbvec-passes="seed-collection<tr-save,bottom-up-vec,tr-accept>" -sbvec-allow-files="" %s -S | FileCheck %s --check-prefix=ALLOW_EMPTY
; RUN: opt -passes=sandbox-vectorizer -sbvec-vec-reg-bits=1024 -sbvec-allow-non-pow2 -sbvec-passes="seed-collection<tr-save,bottom-up-vec,tr-accept>" %s -S | FileCheck %s --check-prefix=DEFAULT

; Checks the command-line option `-sbvec-allow-files`.
define void @widen(ptr %ptr) {
; ALLOW_OTHER:     store float {{%.*}}, ptr {{%.*}}, align 4
; ALLOW_OTHER:     store float {{%.*}}, ptr {{%.*}}, align 4
;
; ALLOW_THIS:      store <2 x float> {{%.*}}, ptr {{%.*}}, align 4
;
; ALLOW_REGEX:     store <2 x float> {{%.*}}, ptr {{%.*}}, align 4
;
; ALLOW_REGEX_CSV: store <2 x float> {{%.*}}, ptr {{%.*}}, align 4
;
; ALLOW_BAD_REGEX: store float {{%.*}}, ptr {{%.*}}, align 4
; ALLOW_BAD_REGEX: store float {{%.*}}, ptr {{%.*}}, align 4
;
; ALLOW_OTHER_CSV: store float {{%.*}}, ptr {{%.*}}, align 4
; ALLOW_OTHER_CSV: store float {{%.*}}, ptr {{%.*}}, align 4
;
; ALLOW_EMPTY:     store float {{%.*}}, ptr {{%.*}}, align 4
; ALLOW_EMPTY:     store float {{%.*}}, ptr {{%.*}}, align 4
;
; DEFAULT:         store <2 x float> {{%.*}}, ptr {{%.*}}, align 4
;
  %ptr0 = getelementptr float, ptr %ptr, i32 0
  %ptr1 = getelementptr float, ptr %ptr, i32 1
  %ld0 = load float, ptr %ptr0
  %ld1 = load float, ptr %ptr1
  store float %ld0, ptr %ptr0
  store float %ld1, ptr %ptr1
  ret void
}
