; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff < %s | \
; RUN:   FileCheck --check-prefix=AIX-64 %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-ibm-aix-xcoff < %s | \
; RUN:   FileCheck --check-prefix=AIX-32 %s

%0 = type { i8*, i8*, i8*, i8*, i8*, i32, i32, i32, i16, i16, [4 x i64] }
%1 = type { [167 x i64] }
%2 = type { [179 x i64] }
%3 = type { i64, void (i32, %3*)*, i64, i64 }

declare i32 @wibble(%1*) local_unnamed_addr #0

declare hidden fastcc i32 @spam(%1*, %2*, %3*) unnamed_addr #0

; Function Attrs: nounwind
define void @baz(%3* %0) local_unnamed_addr #2 {
; AIX-64: std 31
; AIX-64: .byte 0x01 # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 1
; AIX-32: stw 31
; AIX-32: .byte 0x01 # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 1
  %2 = call signext i32 @wibble(%1* nonnull undef) #2
  %3 = call fastcc zeroext i32 @spam(%1* nonnull undef, %2* nonnull undef, %3* nonnull %0)
  unreachable
}
