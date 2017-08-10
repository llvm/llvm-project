; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=ppc32 | FileCheck %s -check-prefix=X32
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=ppc64 | FileCheck %s -check-prefix=X64
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=pwr7 | FileCheck %s -check-prefix=PWR7
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=pwr7 | FileCheck %s -check-prefix=X32


define void @STWBRX(i32 %i, i8* %ptr, i32 %off) {
        %tmp1 = getelementptr i8, i8* %ptr, i32 %off                ; <i8*> [#uses=1]
        %tmp1.upgrd.1 = bitcast i8* %tmp1 to i32*               ; <i32*> [#uses=1]
        %tmp13 = tail call i32 @llvm.bswap.i32( i32 %i )                ; <i32> [#uses=1]
        store i32 %tmp13, i32* %tmp1.upgrd.1
        ret void
}

define i32 @LWBRX(i8* %ptr, i32 %off) {
        %tmp1 = getelementptr i8, i8* %ptr, i32 %off                ; <i8*> [#uses=1]
        %tmp1.upgrd.2 = bitcast i8* %tmp1 to i32*               ; <i32*> [#uses=1]
        %tmp = load i32, i32* %tmp1.upgrd.2          ; <i32> [#uses=1]
        %tmp14 = tail call i32 @llvm.bswap.i32( i32 %tmp )              ; <i32> [#uses=1]
        ret i32 %tmp14
}

define void @STHBRX(i16 %s, i8* %ptr, i32 %off) {
        %tmp1 = getelementptr i8, i8* %ptr, i32 %off                ; <i8*> [#uses=1]
        %tmp1.upgrd.3 = bitcast i8* %tmp1 to i16*               ; <i16*> [#uses=1]
        %tmp5 = call i16 @llvm.bswap.i16( i16 %s )              ; <i16> [#uses=1]
        store i16 %tmp5, i16* %tmp1.upgrd.3
        ret void
}

define i16 @LHBRX(i8* %ptr, i32 %off) {
        %tmp1 = getelementptr i8, i8* %ptr, i32 %off                ; <i8*> [#uses=1]
        %tmp1.upgrd.4 = bitcast i8* %tmp1 to i16*               ; <i16*> [#uses=1]
        %tmp = load i16, i16* %tmp1.upgrd.4          ; <i16> [#uses=1]
        %tmp6 = call i16 @llvm.bswap.i16( i16 %tmp )            ; <i16> [#uses=1]
        ret i16 %tmp6
}

define void @STDBRX(i64 %i, i8* %ptr, i64 %off) {
        %tmp1 = getelementptr i8, i8* %ptr, i64 %off                ; <i8*> [#uses=1]
        %tmp1.upgrd.1 = bitcast i8* %tmp1 to i64*               ; <i64*> [#uses=1]
        %tmp13 = tail call i64 @llvm.bswap.i64( i64 %i )                ; <i64> [#uses=1]
        store i64 %tmp13, i64* %tmp1.upgrd.1
        ret void
}

define i64 @LDBRX(i8* %ptr, i64 %off) {
        %tmp1 = getelementptr i8, i8* %ptr, i64 %off                ; <i8*> [#uses=1]
        %tmp1.upgrd.2 = bitcast i8* %tmp1 to i64*               ; <i64*> [#uses=1]
        %tmp = load i64, i64* %tmp1.upgrd.2          ; <i64> [#uses=1]
        %tmp14 = tail call i64 @llvm.bswap.i64( i64 %tmp )              ; <i64> [#uses=1]
        ret i64 %tmp14
}

declare i32 @llvm.bswap.i32(i32)

declare i16 @llvm.bswap.i16(i16)

declare i64 @llvm.bswap.i64(i64)


; X32: stwbrx
; X32: lwbrx
; X32: sthbrx
; X32: lhbrx
; X32-NOT: ldbrx
; X32-NOT: stdbrx

; X64: stwbrx
; X64: lwbrx
; X64: sthbrx
; X64: lhbrx
; X64-NOT: ldbrx
; X64-NOT: stdbrx

; PWR7: stwbrx
; PWR7: lwbrx
; PWR7: sthbrx
; PWR7: lhbrx
; PWR7: stdbrx
; PWR7: ldbrx

