; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-address-map | FileCheck %s --check-prefixes=CHECK,ELF,ELF-FS
; RUN: llc < %s -mtriple=x86_64 -basic-block-address-map | FileCheck %s --check-prefixes=CHECK,ELF,ELF-NOFS
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc -function-sections -basic-block-address-map | FileCheck %s --check-prefixes=CHECK,COFF-FS
; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc -basic-block-address-map | FileCheck %s --check-prefixes=CHECK,COFF-NOFS

;; GNU COFF (MinGW) does not support associative COMDAT.
; RUN: not --crash llc < %s -mtriple=x86_64-w64-mingw32 -function-sections -basic-block-address-map 2>&1 | FileCheck %s --check-prefix=MINGW-ERR
; MINGW-ERR: BB address map requires associative COMDAT support for COMDAT functions

$_Z4fooTIiET_v = comdat any

define dso_local i32 @_Z3barv() {
  ret i32 0
}
;; For ELF, check we add SHF_LINK_ORDER for .llvm_bb_addr_map and link it with the corresponding .text sections.
;; For COFF, it does not have SHF_LINK_ORDER like mechanism, we use function symbol to "link" them.
; ELF-FS:		.section .text._Z3barv,"ax",@progbits
; COFF-FS:		.section .text,"xr",one_only,_Z3barv
; CHECK-LABEL:	_Z3barv:
; CHECK-NEXT:	[[BAR_BEGIN:.Lfunc_begin[0-9]+]]:
; ELF-FS:		.section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text._Z3barv{{$}}
; ELF-NOFS:		.section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text{{$}}
; COFF-FS:		.section .llvm_bb_addr_map,"drD",associative,_Z3barv,unique,0{{$}}
; COFF-NOFS:		.section .llvm_bb_addr_map,"drD"{{$}}
; CHECK-NEXT:		.byte  5			# version
; CHECK-NEXT:		.short 0			# feature
; CHECK-NEXT:		.quad [[BAR_BEGIN]]	# function address


define dso_local i32 @_Z3foov() {
  %1 = call i32 @_Z4fooTIiET_v()
  ret i32 %1
}
; ELF-FS:		.section .text._Z3foov,"ax",@progbits
; COFF-FS:		.section .text,"xr",one_only,_Z3foov
; CHECK-LABEL:	_Z3foov:
; CHECK-NEXT:	[[FOO_BEGIN:.Lfunc_begin[0-9]+]]:
; ELF-FS:		.section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text._Z3foov{{$}}
; ELF-NOFS:		.section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text{{$}}
; COFF-FS:		.section .llvm_bb_addr_map,"drD",associative,_Z3foov,unique,1{{$}}
; COFF-NOFS:		.section .llvm_bb_addr_map,"drD"{{$}}
; CHECK-NEXT:		.byte  5			# version
; CHECK-NEXT:		.short 32                # feature
; CHECK-NEXT:		.quad [[FOO_BEGIN]]	# function address


define linkonce_odr dso_local i32 @_Z4fooTIiET_v() comdat {
  ret i32 0
}
;; Check we add .llvm_bb_addr_map section to a COMDAT group with the corresponding .text section if such a COMDAT exists.
; ELF:			.section .text._Z4fooTIiET_v,"axG",@progbits,_Z4fooTIiET_v,comdat
; COFF-FS:		.section .text,"xr",discard,_Z4fooTIiET_v,unique,2{{$}}
; COFF-NOFS:		.section .text,"xr",discard,_Z4fooTIiET_v{{$}}
; CHECK-LABEL:	_Z4fooTIiET_v:
; CHECK-NEXT:	[[FOOCOMDAT_BEGIN:.Lfunc_begin[0-9]+]]:
; ELF:			.section .llvm_bb_addr_map,"oG",@llvm_bb_addr_map,.text._Z4fooTIiET_v,_Z4fooTIiET_v,comdat{{$}}
; COFF-FS:		.section .llvm_bb_addr_map,"drD",associative,_Z4fooTIiET_v,unique,2{{$}}
; COFF-NOFS:		.section .llvm_bb_addr_map,"drD",associative,_Z4fooTIiET_v{{$}}
; CHECK-NEXT:		.byte  5				# version
; CHECK-NEXT:		.short 0				# feature
; CHECK-NEXT:		.quad [[FOOCOMDAT_BEGIN]]	# function address
