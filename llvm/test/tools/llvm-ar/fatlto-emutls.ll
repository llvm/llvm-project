;; A fat LTO object (ELF or COFF) carries its bitcode in a .llvm.lto section,
;; and a linker that links the bitcode looks the bitcode names up in the archive
;; map. With emulated TLS a thread-local variable X is __emutls_v.X in the
;; native code but X in the bitcode, so the map has to list the names of both.
;; The names the two have in common are listed once.

; REQUIRES: x86-registered-target

; RUN: rm -rf %t && mkdir %t && cd %t

; RUN: opt -mtriple=x86_64-linux-gnu -passes='fatlto-pre-link<O2>' %s | llc -emulated-tls --filetype=obj -o elf.o
; RUN: llvm-ar rcs elf.a elf.o
; RUN: llvm-nm --print-armap elf.a | FileCheck %s --check-prefix=ELF

; RUN: opt -mtriple=i686-w64-windows-gnu -passes='fatlto-pre-link<O2>' %s | llc -emulated-tls --filetype=obj -o coff.o
; RUN: llvm-ar rcs coff.a coff.o
; RUN: llvm-nm --print-armap coff.a | FileCheck %s --check-prefix=COFF
; RUN: llvm-ar --format=gnu rcs coff-gnu.a coff.o
; RUN: llvm-nm --print-armap coff-gnu.a | FileCheck %s --check-prefix=COFF-GNU

;; Without emulated TLS the two have the same names, and the map is as before.
; RUN: opt -mtriple=i686-w64-windows-gnu -passes='fatlto-pre-link<O2>' %s | llc --filetype=obj -o native.o
; RUN: llvm-ar rcs native.a native.o
; RUN: llvm-nm --print-armap native.a | FileCheck %s --check-prefix=NATIVE

;; The bitcode embedded in a Mach-O object (-fembed-bitcode) is not fat LTO and
;; is not indexed.
; RUN: opt -mtriple=x86_64-apple-macosx %s -o macho.bc
; RUN: llc -mtriple=x86_64-apple-macosx -emulated-tls --filetype=obj %s -o macho.o
; RUN: llvm-objcopy --add-section=__LLVM,__bitcode=macho.bc macho.o
; RUN: llvm-ar rcs macho.a macho.o
; RUN: llvm-nm --print-armap macho.a | FileCheck %s --check-prefix=MACHO

; ELF:      Archive map
; ELF-NEXT: get in elf.o
; ELF-NEXT: __emutls_v.tls_zero in elf.o
; ELF-NEXT: __emutls_v.tls_init in elf.o
; ELF-NEXT: __emutls_t.tls_init in elf.o
; ELF-NEXT: tls_zero in elf.o
; ELF-NEXT: tls_init in elf.o
; ELF-EMPTY:

; COFF:      Archive map
; COFF-NEXT: ___emutls_t.tls_init in coff.o
; COFF-NEXT: ___emutls_v.tls_init in coff.o
; COFF-NEXT: ___emutls_v.tls_zero in coff.o
; COFF-NEXT: _get in coff.o
; COFF-NEXT: _tls_init in coff.o
; COFF-NEXT: _tls_zero in coff.o
; COFF-EMPTY:

; COFF-GNU:      Archive map
; COFF-GNU-NEXT: _get in coff.o
; COFF-GNU-NEXT: ___emutls_v.tls_zero in coff.o
; COFF-GNU-NEXT: ___emutls_v.tls_init in coff.o
; COFF-GNU-NEXT: ___emutls_t.tls_init in coff.o
; COFF-GNU-NEXT: _tls_zero in coff.o
; COFF-GNU-NEXT: _tls_init in coff.o
; COFF-GNU-EMPTY:

; NATIVE:      Archive map
; NATIVE-NEXT: _get in native.o
; NATIVE-NEXT: _tls_init in native.o
; NATIVE-NEXT: _tls_zero in native.o
; NATIVE-EMPTY:

; MACHO:      Archive map
; MACHO-NEXT: ___emutls_t.tls_init in macho.o
; MACHO-NEXT: ___emutls_v.tls_init in macho.o
; MACHO-NEXT: ___emutls_v.tls_zero in macho.o
; MACHO-NEXT: _get in macho.o
; MACHO-EMPTY:

@tls_zero = thread_local global i32 0, align 4
@tls_init = thread_local global i32 1, align 4

define i32 @get() {
  %zero = load i32, ptr @tls_zero, align 4
  %init = load i32, ptr @tls_init, align 4
  %sum = add i32 %zero, %init
  ret i32 %sum
}
