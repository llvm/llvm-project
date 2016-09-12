; RUN: llc -mtriple=aarch64-linux-gnu -O0 -global-isel -stop-after=irtranslator -o - %s | FileCheck %s

%type = type [4 x {i8, i32}]

define %type* @first_offset_const(%type* %addr) {
; CHECK-LABEL: name: first_offset_const
; CHECK: [[BASE:%[0-9]+]](p0) = COPY %x0
; CHECK: [[OFFSET:%[0-9]+]](s64) = G_CONSTANT 32
; CHECK: [[RES:%[0-9]+]](p0) = G_GEP [[BASE]], [[OFFSET]](s64)
; CHECK: %x0 = COPY [[RES]](p0)

  %res = getelementptr %type, %type* %addr, i32 1
  ret %type* %res
}

define %type* @first_offset_trivial(%type* %addr) {
; CHECK-LABEL: name: first_offset_trivial
; CHECK: [[BASE:%[0-9]+]](p0) = COPY %x0
; CHECK: [[TRIVIAL:%[0-9]+]](p0) = COPY [[BASE]](p0)
; CHECK: %x0 = COPY [[TRIVIAL]](p0)

  %res = getelementptr %type, %type* %addr, i32 0
  ret %type* %res
}

define %type* @first_offset_variable(%type* %addr, i64 %idx) {
; CHECK-LABEL: name: first_offset_variable
; CHECK: [[BASE:%[0-9]+]](p0) = COPY %x0
; CHECK: [[IDX:%[0-9]+]](s64) = COPY %x1
; CHECK: [[SIZE:%[0-9]+]](s64) = G_CONSTANT 32
; CHECK: [[OFFSET:%[0-9]+]](s64) = G_MUL [[SIZE]], [[IDX]]
; CHECK: [[STEP0:%[0-9]+]](p0) = G_GEP [[BASE]], [[OFFSET]](s64)
; CHECK: [[RES:%[0-9]+]](p0) = COPY [[STEP0]](p0)
; CHECK: %x0 = COPY [[RES]](p0)

  %res = getelementptr %type, %type* %addr, i64 %idx
  ret %type* %res
}

define %type* @first_offset_ext(%type* %addr, i32 %idx) {
; CHECK-LABEL: name: first_offset_ext
; CHECK: [[BASE:%[0-9]+]](p0) = COPY %x0
; CHECK: [[IDX32:%[0-9]+]](s32) = COPY %w1
; CHECK: [[SIZE:%[0-9]+]](s64) = G_CONSTANT 32
; CHECK: [[IDX64:%[0-9]+]](s64) = G_SEXT [[IDX32]](s32)
; CHECK: [[OFFSET:%[0-9]+]](s64) = G_MUL [[SIZE]], [[IDX64]]
; CHECK: [[STEP0:%[0-9]+]](p0) = G_GEP [[BASE]], [[OFFSET]](s64)
; CHECK: [[RES:%[0-9]+]](p0) = COPY [[STEP0]](p0)
; CHECK: %x0 = COPY [[RES]](p0)

  %res = getelementptr %type, %type* %addr, i32 %idx
  ret %type* %res
}

%type1 = type [4 x [4 x i32]]
define i32* @const_then_var(%type1* %addr, i64 %idx) {
; CHECK-LABEL: name: const_then_var
; CHECK: [[BASE:%[0-9]+]](p0) = COPY %x0
; CHECK: [[IDX:%[0-9]+]](s64) = COPY %x1
; CHECK: [[OFFSET1:%[0-9]+]](s64) = G_CONSTANT 272
; CHECK: [[BASE1:%[0-9]+]](p0) = G_GEP [[BASE]], [[OFFSET1]](s64)
; CHECK: [[SIZE:%[0-9]+]](s64) = G_CONSTANT 4
; CHECK: [[OFFSET2:%[0-9]+]](s64) = G_MUL [[SIZE]], [[IDX]]
; CHECK: [[BASE2:%[0-9]+]](p0) = G_GEP [[BASE1]], [[OFFSET2]](s64)
; CHECK: [[RES:%[0-9]+]](p0) = COPY [[BASE2]](p0)
; CHECK: %x0 = COPY [[RES]](p0)

  %res = getelementptr %type1, %type1* %addr, i32 4, i32 1, i64 %idx
  ret i32* %res
}

define i32* @var_then_const(%type1* %addr, i64 %idx) {
; CHECK-LABEL: name: var_then_const
; CHECK: [[BASE:%[0-9]+]](p0) = COPY %x0
; CHECK: [[IDX:%[0-9]+]](s64) = COPY %x1
; CHECK: [[SIZE:%[0-9]+]](s64) = G_CONSTANT 64
; CHECK: [[OFFSET1:%[0-9]+]](s64) = G_MUL [[SIZE]], [[IDX]]
; CHECK: [[BASE1:%[0-9]+]](p0) = G_GEP [[BASE]], [[OFFSET1]](s64)
; CHECK: [[OFFSET2:%[0-9]+]](s64) = G_CONSTANT 40
; CHECK: [[BASE2:%[0-9]+]](p0) = G_GEP [[BASE1]], [[OFFSET2]](s64)
; CHECK: %x0 = COPY [[BASE2]](p0)

  %res = getelementptr %type1, %type1* %addr, i64 %idx, i32 2, i32 2
  ret i32* %res
}
