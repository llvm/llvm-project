; Test that the msan layout customization options work as expected
;
; RUN: opt < %s -msan-shadow-base 3735928559 -S -passes=msan 2>&1 | FileCheck  \
; RUN: --check-prefix=CHECK-BASE %s
; RUN: opt < %s -msan -msan-shadow-base 3735928559 -S | FileCheck --check-prefix=CHECK-BASE %s
; RUN: opt < %s -msan-shadow-base 3735928559 -msan-and-mask 4294901760 -S      \
; RUN: -passes=msan 2>&1 | FileCheck --check-prefix=CHECK-AND %s
; RUN: opt < %s -msan -msan-shadow-base 3735928559 -msan-and-mask 4294901760 -S | FileCheck --check-prefix=CHECK-AND %s
; RUN: opt < %s -msan-shadow-base 3735928559 -msan-xor-mask 48879 -S           \
; RUN: -passes=msan 2>&1 | FileCheck --check-prefix=CHECK-XOR %s
; RUN: opt < %s -msan -msan-shadow-base 3735928559 -msan-xor-mask 48879 -S | FileCheck --check-prefix=CHECK-XOR %s
; RUN: opt < %s -msan-shadow-base 3735928559 -msan-xor-mask 48879              \
; RUN: -msan-and-mask 4294901760 -S -passes=msan 2>&1 | FileCheck              \
; RUN: --check-prefix=CHECK-XOR-AND %s
; RUN: opt < %s -msan -msan-shadow-base 3735928559 -msan-xor-mask 48879 -msan-and-mask 4294901760 -S | FileCheck --check-prefix=CHECK-XOR-AND %s
; RUN: opt < %s -msan-track-origins 1 -msan-origin-base 1777777 -S -passes=msan\
; RUN: 2>&1 | FileCheck --check-prefix=CHECK-ORIGIN-BASE %s
; RUN: opt < %s -msan -msan-track-origins 1 -msan-origin-base 1777777 -S | FileCheck --check-prefix=CHECK-ORIGIN-BASE %s

target triple = "x86_64-unknown-linux-gnu"

define i32 @read_value(i32* %a) sanitize_memory {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
; CHECK-BASE-LABEL: @read_value
; CHECK-BASE-NOT: ret i32
; CHECK-BASE: add{{.*}}3735928559
; CHECK-BASE: ret i32

; CHECK-AND-LABEL: @read_value
; CHECK-AND-NOT: ret i32
; CHECK-AND: and{{.*}}-4294901761
; CHECK-AND-NEXT: add{{.*}}3735928559
; CHECK-AND: ret i32

; CHECK-XOR-LABEL: @read_value
; CHECK-XOR-NOT: ret i32
; CHECK-XOR: xor{{.*}}48879
; CHECK-XOR-NEXT: add{{.*}}3735928559
; CHECK-XOR: ret i32

; CHECK-XOR-AND-LABEL: @read_value
; CHECK-XOR-AND-NOT: ret i32
; CHECK-XOR-AND: and{{.*}}-4294901761
; CHECK-XOR-AND-NEXT: xor{{.*}}48879
; CHECK-XOR-AND-NEXT: add{{.*}}3735928559
; CHECK-XOR-AND: ret i32

; CHECK-ORIGIN-BASE-LABEL: @read_value
; CHECK-ORIGIN-BASE-NOT: ret i32
; CHECK-ORIGIN-BASE: add{{.*}}1777777
; CHECK-ORIGIN-BASE: ret i32
