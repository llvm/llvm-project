; RUN: llc -mtriple=i386-unknown-unknown -mcpu=core-avx2 < %s | FileCheck --check-prefix=BMI2 %s
; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=core-avx2 < %s | FileCheck --check-prefix=BMI264 %s

define i32 @shl32(i32 %x, i32 %shamt) nounwind uwtable readnone {
entry:
  %shl = shl i32 %x, %shamt
; BMI2: shl32
; BMI2: shlxl
; BMI2: ret
; BMI264: shl32
; BMI264: shlxl
; BMI264: ret
  ret i32 %shl
}

define i32 @shl32i(i32 %x) nounwind uwtable readnone {
entry:
  %shl = shl i32 %x, 5
; BMI2: shl32i
; BMI2-NOT: shlxl
; BMI2: ret
; BMI264: shl32i
; BMI264-NOT: shlxl
; BMI264: ret
  ret i32 %shl
}

define i32 @shl32p(i32* %p, i32 %shamt) nounwind uwtable readnone {
entry:
  %x = load i32, i32* %p
  %shl = shl i32 %x, %shamt
; BMI2: shl32p
; BMI2: shlxl %{{.+}}, ({{.+}}), %{{.+}}
; BMI2: ret
; BMI264: shl32p
; BMI264: shlxl %{{.+}}, ({{.+}}), %{{.+}}
; BMI264: ret
  ret i32 %shl
}

define i32 @shl32pi(i32* %p) nounwind uwtable readnone {
entry:
  %x = load i32, i32* %p
  %shl = shl i32 %x, 5
; BMI2: shl32pi
; BMI2-NOT: shlxl
; BMI2: ret
; BMI264: shl32pi
; BMI264-NOT: shlxl
; BMI264: ret
  ret i32 %shl
}

define i64 @shl64(i64 %x, i64 %shamt) nounwind uwtable readnone {
entry:
  %shl = shl i64 %x, %shamt
; BMI264: shl64
; BMI264: shlxq
; BMI264: ret
  ret i64 %shl
}

define i64 @shl64i(i64 %x) nounwind uwtable readnone {
entry:
  %shl = shl i64 %x, 7
; BMI264: shl64i
; BMI264-NOT: shlxq
; BMI264: ret
  ret i64 %shl
}

define i64 @shl64p(i64* %p, i64 %shamt) nounwind uwtable readnone {
entry:
  %x = load i64, i64* %p
  %shl = shl i64 %x, %shamt
; BMI264: shl64p
; BMI264: shlxq %{{.+}}, ({{.+}}), %{{.+}}
; BMI264: ret
  ret i64 %shl
}

define i64 @shl64pi(i64* %p) nounwind uwtable readnone {
entry:
  %x = load i64, i64* %p
  %shl = shl i64 %x, 7
; BMI264: shl64pi
; BMI264-NOT: shlxq
; BMI264: ret
  ret i64 %shl
}

define i32 @lshr32(i32 %x, i32 %shamt) nounwind uwtable readnone {
entry:
  %shl = lshr i32 %x, %shamt
; BMI2: lshr32
; BMI2: shrxl
; BMI2: ret
; BMI264: lshr32
; BMI264: shrxl
; BMI264: ret
  ret i32 %shl
}

define i32 @lshr32p(i32* %p, i32 %shamt) nounwind uwtable readnone {
entry:
  %x = load i32, i32* %p
  %shl = lshr i32 %x, %shamt
; BMI2: lshr32p
; BMI2: shrxl %{{.+}}, ({{.+}}), %{{.+}}
; BMI2: ret
; BMI264: lshr32p
; BMI264: shrxl %{{.+}}, ({{.+}}), %{{.+}}
; BMI264: ret
  ret i32 %shl
}

define i64 @lshr64(i64 %x, i64 %shamt) nounwind uwtable readnone {
entry:
  %shl = lshr i64 %x, %shamt
; BMI264: lshr64
; BMI264: shrxq
; BMI264: ret
  ret i64 %shl
}

define i64 @lshr64p(i64* %p, i64 %shamt) nounwind uwtable readnone {
entry:
  %x = load i64, i64* %p
  %shl = lshr i64 %x, %shamt
; BMI264: lshr64p
; BMI264: shrxq %{{.+}}, ({{.+}}), %{{.+}}
; BMI264: ret
  ret i64 %shl
}

define i32 @ashr32(i32 %x, i32 %shamt) nounwind uwtable readnone {
entry:
  %shl = ashr i32 %x, %shamt
; BMI2: ashr32
; BMI2: sarxl
; BMI2: ret
; BMI264: ashr32
; BMI264: sarxl
; BMI264: ret
  ret i32 %shl
}

define i32 @ashr32p(i32* %p, i32 %shamt) nounwind uwtable readnone {
entry:
  %x = load i32, i32* %p
  %shl = ashr i32 %x, %shamt
; BMI2: ashr32p
; Source order scheduling prevents folding, rdar:14208996.
; BMI2: sarxl %{{.+}}, ({{.+}}), %{{.+}}
; BMI2: ret
; BMI264: ashr32p
; BMI264: sarxl %{{.+}}, ({{.+}}), %{{.+}}
; BMI264: ret
  ret i32 %shl
}

define i64 @ashr64(i64 %x, i64 %shamt) nounwind uwtable readnone {
entry:
  %shl = ashr i64 %x, %shamt
; BMI264: ashr64
; BMI264: sarxq
; BMI264: ret
  ret i64 %shl
}

define i64 @ashr64p(i64* %p, i64 %shamt) nounwind uwtable readnone {
entry:
  %x = load i64, i64* %p
  %shl = ashr i64 %x, %shamt
; BMI264: ashr64p
; BMI264: sarxq %{{.+}}, ({{.+}}), %{{.+}}
; BMI264: ret
  ret i64 %shl
}
