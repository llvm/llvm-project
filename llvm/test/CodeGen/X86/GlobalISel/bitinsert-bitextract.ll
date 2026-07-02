; RUN: llc -mtriple=x86_64-unknown-linux-gnu -global-isel -stop-after=irtranslator < %s | FileCheck %s
define i16 @test_bitextract(b32 %src, i32 %off) {
; CHECK-LABEL: name: test_bitextract
; CHECK: [[SRC:%[0-9]+]]:_(s32) = COPY $edi
; CHECK: [[OFF:%[0-9]+]]:_(s32) = COPY $esi
; CHECK: [[OFF2:%[0-9]+]]:_(s32) = COPY [[OFF]](s32)
; CHECK: [[SHR:%[0-9]+]]:_(s32) = G_LSHR [[SRC]], [[OFF2]](s32)
; CHECK: [[RES:%[0-9]+]]:_(s16) = G_TRUNC [[SHR]](s32)
; CHECK: $ax = COPY [[RES]](s16)
; CHECK: RET 0, implicit $ax
  %result = bitextract i16, b32 %src, i32 %off
  ret i16 %result
}
define b32 @test_bitinsert(b32 %base, i16 %val, i32 %off) {
; CHECK-LABEL: name: test_bitinsert
; CHECK: [[BASE:%[0-9]+]]:_(s32) = COPY $edi
; CHECK: [[VAL:%[0-9]+]]:_(s16) = G_TRUNC %{{[0-9]+}}(s32)
; CHECK: [[OFF:%[0-9]+]]:_(s32) = COPY $edx
; CHECK: [[OFF2:%[0-9]+]]:_(s32) = COPY [[OFF]](s32)
; CHECK: [[ROTR:%[0-9]+]]:_(s32) = G_ROTR [[BASE]], [[OFF2]](s32)
; CHECK: [[EXT_VAL:%[0-9]+]]:_(s32) = G_ZEXT [[VAL]](s16)
; CHECK: [[MASK:%[0-9]+]]:_(s32) = G_CONSTANT i32 -65536
; CHECK: [[AND:%[0-9]+]]:_(s32) = G_AND [[ROTR]], [[MASK]]
; CHECK: [[OR:%[0-9]+]]:_(s32) = G_OR [[AND]], [[EXT_VAL]]
; CHECK: [[ROTL:%[0-9]+]]:_(s32) = G_ROTL [[OR]], [[OFF2]](s32)
; CHECK: $eax = COPY [[ROTL]](s32)
; CHECK: RET 0, implicit $eax
  %result = bitinsert b32 %base, i16 %val, i32 %off
  ret b32 %result
}