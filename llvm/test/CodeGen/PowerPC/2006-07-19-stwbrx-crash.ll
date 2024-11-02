; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--

define void @img2buf(i32 %symbol_size_in_bytes, ptr %ui16) nounwind {
        %tmp93 = load i16, ptr null         ; <i16> [#uses=1]
        %tmp99 = call i16 @llvm.bswap.i16( i16 %tmp93 )         ; <i16> [#uses=1]
        store i16 %tmp99, ptr %ui16
        ret void
}

declare i16 @llvm.bswap.i16(i16)

