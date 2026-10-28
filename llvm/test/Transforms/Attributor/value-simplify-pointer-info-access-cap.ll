; NOTE: Tests that -attributor-max-pi-accesses caps AAPointerInfo and prevents
; value simplification when the access limit is exceeded.

; Without the cap, the Attributor simplifies loads via AAPointerInfo.
; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal  -attributor-annotate-decl-cs  -S < %s | FileCheck %s --check-prefix=NOCAP
; RUN: opt -aa-pipeline=basic-aa -passes=attributor-cgscc -attributor-manifest-internal  -attributor-annotate-decl-cs -S < %s | FileCheck %s --check-prefix=NOCAP

; With cap=3, AAPointerInfo goes pessimistic after 3 accesses and
; value simplification is blocked.
; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal  -attributor-annotate-decl-cs  -attributor-max-pi-accesses=3 -S < %s | FileCheck %s --check-prefix=CAP
; RUN: opt -aa-pipeline=basic-aa -passes=attributor-cgscc -attributor-manifest-internal  -attributor-annotate-decl-cs -attributor-max-pi-accesses=3 -S < %s | FileCheck %s --check-prefix=CAP

; Verify that the cap triggers the expected statistic.
; REQUIRES: asserts
; RUN: opt -aa-pipeline=basic-aa -passes=attributor -attributor-manifest-internal  -attributor-annotate-decl-cs  -attributor-max-pi-accesses=3 -stats -S < %s 2>&1 | FileCheck %s --check-prefix=STATS


; Test 1: Alloca with multiple stores followed by a load.
; Without the cap, the load is simplified to the last stored value.
; With cap=3, AAPointerInfo for %p goes pessimistic after 3 accesses
; and the load cannot be simplified.

define i32 @local_alloca_simplifiable() {
; NOCAP-LABEL: define {{.*}}i32 @local_alloca_simplifiable
; NOCAP-NEXT:    ret i32 5
;
; CAP-LABEL: define {{.*}}i32 @local_alloca_simplifiable
; CAP:         %v = load i32, ptr %p
; CAP-NEXT:    ret i32 %v
  %p = alloca i32, align 4
  store i32 1, ptr %p, align 4
  store i32 2, ptr %p, align 4
  store i32 3, ptr %p, align 4
  store i32 4, ptr %p, align 4
  store i32 5, ptr %p, align 4
  %v = load i32, ptr %p, align 4
  ret i32 %v
}


; Test 2: Inter-procedural case with an internal global.
; Four separate functions write the same value (42) to @G.
; Without the cap, the Attributor determines the unique stored value
; and simplifies the load in @reader to ret i32 42.
; With cap=3, the global's AAPointerInfoFloating goes pessimistic
; after 3 accesses and the load cannot be simplified.

@G = internal global i32 poison, align 4

define void @writer1() {
  store i32 42, ptr @G, align 4
  ret void
}

define void @writer2() {
  store i32 42, ptr @G, align 4
  ret void
}

define void @writer3() {
  store i32 42, ptr @G, align 4
  ret void
}

define void @writer4() {
  store i32 42, ptr @G, align 4
  ret void
}

define i32 @reader() {
; NOCAP-LABEL: define {{.*}}i32 @reader
; NOCAP:         ret i32 42
;
; CAP-LABEL: define {{.*}}i32 @reader
; CAP:         %v = load i32, ptr @G
; CAP-NEXT:    ret i32 %v
  %v = load i32, ptr @G, align 4
  ret i32 %v
}

; STATS: {{[1-9][0-9]*}} attributor - Number of AAPointerInfo instances capped to pessimistic
