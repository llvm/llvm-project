; RUN: llc -verify-machineinstrs -mtriple=powerpc64-ibm-aix-xcoff < %s | \
; RUN:   FileCheck --check-prefix=AIX-64 %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-ibm-aix-xcoff < %s | \
; RUN:   FileCheck --check-prefix=AIX-32 %s

%0 = type { ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i16, i16, [4 x i64] }
%1 = type { [167 x i64] }
%2 = type { [179 x i64] }
%3 = type { i64, ptr, i64, i64 }

declare i32 @wibble(ptr) local_unnamed_addr #0

declare hidden fastcc i32 @spam(ptr, ptr, ptr) unnamed_addr #0

; Function Attrs: nounwind
define void @baz(ptr %0) local_unnamed_addr #2 {
; AIX-64: std 31
; AIX-64: .byte 0x01 # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 1
; AIX-32: stw 31
; AIX-32: .byte 0x01 # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 1
  %2 = call signext i32 @wibble(ptr nonnull undef) #2
  %3 = call fastcc zeroext i32 @spam(ptr nonnull undef, ptr nonnull undef, ptr nonnull %0)
  unreachable
}
