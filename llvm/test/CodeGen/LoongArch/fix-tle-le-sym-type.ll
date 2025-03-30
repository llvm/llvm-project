; RUN: llc --mtriple=loongarch32 --filetype=obj %s -o %t-la32
; RUN: llvm-readelf -s %t-la32 | FileCheck %s --check-prefix=LA32

; RUN: llc --mtriple=loongarch64 --filetype=obj %s -o %t-la64
; RUN: llvm-readelf -s %t-la64 | FileCheck %s --check-prefix=LA64

; LA32:      Symbol table '.symtab' contains [[#]] entries:
; LA32-NEXT:    Num:    Value  Size Type  Bind   Vis      Ndx Name
; LA32:              00000000     0 TLS   GLOBAL DEFAULT  UND tls_sym

; LA64:      Symbol table '.symtab' contains [[#]] entries:
; LA64-NEXT:    Num:    Value          Size Type  Bind   Vis      Ndx Name
; LA64:              0000000000000000     0 TLS   GLOBAL DEFAULT  UND tls_sym

@tls_sym = external thread_local(localexec) global i32

define dso_local signext i32 @test_tlsle() nounwind {
entry:
  %0 = call ptr @llvm.threadlocal.address.p0(ptr @tls_sym)
  %1 = load i32, ptr %0
  ret i32 %1
}

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)
