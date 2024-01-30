; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false --code-model=large -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefix=RELOC %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYM %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS %s

@TIInitIE = internal thread_local(initialexec) global i32 42, align 4
@TIUninitIE = internal thread_local(initialexec) global i32 0, align 4
@TIInitLD = internal thread_local(localdynamic) global i32 42, align 4
@TIUninitLD = internal thread_local(localdynamic) global i32 0, align 4

define i32 @loadTIInitIE_USE_TLS_LD() #0 {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitIE)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIInitIE_USE_TLS_LD(i32 noundef signext %i) #0 {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitIE)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIUninitIE_USE_TLS_LD() #0 {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitIE)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIUninitIE_USE_TLS_LD(i32 noundef signext %i) #0 {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitIE)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIInitIE() {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitIE)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIInitIE(i32 noundef signext %i) {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitIE)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIUninitIE() {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitIE)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIUninitIE(i32 noundef signext %i) {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitIE)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIInitLD_USE_TLS_IE() #1 {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitLD)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIInitLD_USE_TLS_IE(i32 noundef signext %i) #1 {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitLD)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIUninitLD_USE_TLS_IE() #1 {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitLD)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIUninitLD_USE_TLS_IE(i32 noundef signext %i) #1 {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitLD)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIInitLD() {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitLD)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIInitLD(i32 noundef signext %i) {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIInitLD)
  store i32 %i, ptr %0, align 4
  ret void
}

define i32 @loadTIUninitLD() {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitLD)
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

define void @storeTIUninitLD(i32 noundef signext %i) {
entry:
  %0 = tail call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TIUninitLD)
  store i32 %i, ptr %0, align 4
  ret void
}

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)

attributes #0 = { "target-features"="+aix-func-use-tls-local-dynamic" }

attributes #1 = { "target-features"="+aix-func-use-tls-initial-exec" }

; RELOC:      File: {{.*}}aix-func-use-tls-ld-ie-mix-reloc-large.ll.tmp.o
; RELOC-NEXT: Format: aix5coff64-rs6000
; RELOC-NEXT: Arch: powerpc64
; RELOC-NEXT: AddressSize: 64bit
; RELOC-NEXT: Relocations [
; RELOC:      Virtual Address: 0xA
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0xE
; RELOC-NEXT:      Symbol: TIInitIE (73)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x16
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x18
; RELOC-NEXT:      Symbol: .__tls_get_mod (1)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x1E
; RELOC-NEXT:      Symbol: TIInitIE (73)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x4E
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x52
; RELOC-NEXT:      Symbol: TIInitIE (73)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x5A
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x5C
; RELOC-NEXT:      Symbol: .__tls_get_mod (1)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x62
; RELOC-NEXT:      Symbol: TIInitIE (73)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x8A
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x8E
; RELOC-NEXT:      Symbol: TIUninitIE (75)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x96
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x98
; RELOC-NEXT:      Symbol: .__tls_get_mod (1)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x9E
; RELOC-NEXT:      Symbol: TIUninitIE (75)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0xCE
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0xD2
; RELOC-NEXT:      Symbol: TIUninitIE (75)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0xDA
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0xDC
; RELOC-NEXT:      Symbol: .__tls_get_mod (1)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0xE2
; RELOC-NEXT:      Symbol: TIUninitIE (75)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x102
; RELOC-NEXT:      Symbol: TIInitIE (77)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x106
; RELOC-NEXT:      Symbol: TIInitIE (77)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x112
; RELOC-NEXT:      Symbol: TIInitIE (77)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x116
; RELOC-NEXT:      Symbol: TIInitIE (77)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x122
; RELOC-NEXT:      Symbol: TIUninitIE (79)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x126
; RELOC-NEXT:      Symbol: TIUninitIE (79)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x132
; RELOC-NEXT:      Symbol: TIUninitIE (79)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x136
; RELOC-NEXT:      Symbol: TIUninitIE (79)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x142
; RELOC-NEXT:      Symbol: TIInitLD (81)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x146
; RELOC-NEXT:      Symbol: TIInitLD (81)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x152
; RELOC-NEXT:      Symbol: TIInitLD (81)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x156
; RELOC-NEXT:      Symbol: TIInitLD (81)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x162
; RELOC-NEXT:      Symbol: TIUninitLD (83)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x166
; RELOC-NEXT:      Symbol: TIUninitLD (83)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x172
; RELOC-NEXT:      Symbol: TIUninitLD (83)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x176
; RELOC-NEXT:      Symbol: TIUninitLD (83)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x18A
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x18E
; RELOC-NEXT:      Symbol: TIInitLD (85)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x196
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x198
; RELOC-NEXT:      Symbol: .__tls_get_mod (1)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x19E
; RELOC-NEXT:      Symbol: TIInitLD (85)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x1CE
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x1D2
; RELOC-NEXT:      Symbol: TIInitLD (85)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x1DA
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x1DC
; RELOC-NEXT:      Symbol: .__tls_get_mod (1)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x1E2
; RELOC-NEXT:      Symbol: TIInitLD (85)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x20A
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x20E
; RELOC-NEXT:      Symbol: TIUninitLD (87)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x216
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x218
; RELOC-NEXT:      Symbol: .__tls_get_mod (1)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x21E
; RELOC-NEXT:      Symbol: TIUninitLD (87)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x24E
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x252
; RELOC-NEXT:      Symbol: TIUninitLD (87)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCU (0x30)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x25A
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x25C
; RELOC-NEXT:      Symbol: .__tls_get_mod (1)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 26
; RELOC-NEXT:      Type: R_RBA (0x18)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x262
; RELOC-NEXT:      Symbol: TIUninitLD (87)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 16
; RELOC-NEXT:      Type: R_TOCL (0x31)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x278
; RELOC-NEXT:      Symbol: .loadTIInitIE_USE_TLS_LD (5)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x280
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x290
; RELOC-NEXT:      Symbol: .storeTIInitIE_USE_TLS_LD (7)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x298
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x2A8
; RELOC-NEXT:      Symbol: .loadTIUninitIE_USE_TLS_LD (9)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x2B0
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x2C0
; RELOC-NEXT:      Symbol: .storeTIUninitIE_USE_TLS_LD (11)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x2C8
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x2D8
; RELOC-NEXT:      Symbol: .loadTIInitIE (13)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x2E0
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x2F0
; RELOC-NEXT:      Symbol: .storeTIInitIE (15)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x2F8
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x308
; RELOC-NEXT:      Symbol: .loadTIUninitIE (17)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x310
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x320
; RELOC-NEXT:      Symbol: .storeTIUninitIE (19)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x328
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x338
; RELOC-NEXT:      Symbol: .loadTIInitLD_USE_TLS_IE (21)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x340
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x350
; RELOC-NEXT:      Symbol: .storeTIInitLD_USE_TLS_IE (23)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x358
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x368
; RELOC-NEXT:      Symbol: .loadTIUninitLD_USE_TLS_IE (25)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x370
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x380
; RELOC-NEXT:      Symbol: .storeTIUninitLD_USE_TLS_IE (27)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x388
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x398
; RELOC-NEXT:      Symbol: .loadTIInitLD (29)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x3A0
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x3B0
; RELOC-NEXT:      Symbol: .storeTIInitLD (31)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x3B8
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x3C8
; RELOC-NEXT:      Symbol: .loadTIUninitLD (33)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x3D0
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x3E0
; RELOC-NEXT:      Symbol: .storeTIUninitLD (35)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x3E8
; RELOC-NEXT:      Symbol: TOC (69)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_POS (0x0)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x3F8
; RELOC-NEXT:      Symbol: _$TLSML (71)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLSML (0x25)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x400
; RELOC-NEXT:      Symbol: TIInitIE (89)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_LD (0x22)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x408
; RELOC-NEXT:      Symbol: TIUninitIE (93)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_LD (0x22)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x410
; RELOC-NEXT:      Symbol: TIInitIE (89)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_IE (0x21)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x418
; RELOC-NEXT:      Symbol: TIUninitIE (93)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_IE (0x21)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x420
; RELOC-NEXT:      Symbol: TIInitLD (91)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_IE (0x21)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x428
; RELOC-NEXT:      Symbol: TIUninitLD (95)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_IE (0x21)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x430
; RELOC-NEXT:      Symbol: TIInitLD (91)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_LD (0x22)
; RELOC-NEXT:    }
; RELOC:      Virtual Address: 0x438
; RELOC-NEXT:      Symbol: TIUninitLD (95)
; RELOC-NEXT:      IsSigned: No
; RELOC-NEXT:      FixupBitValue: 0
; RELOC-NEXT:      Length: 64
; RELOC-NEXT:      Type: R_TLS_LD (0x22)
; RELOC-NEXT:    }

; SYM:      File: {{.*}}aix-func-use-tls-ld-ie-mix-reloc-large.ll.tmp.o
; SYM-NEXT: Format: aix5coff64-rs6000
; SYM-NEXT: Arch: powerpc64
; SYM-NEXT: AddressSize: 64bit
; SYM-NEXT: Symbols [
; SYM:    Index: 71
; SYM-NEXT:    Name: _$TLSML
; SYM-NEXT:    Value (RelocatableAddress): 0x3F8
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 72
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 73
; SYM-NEXT:    Name: TIInitIE
; SYM-NEXT:    Value (RelocatableAddress): 0x400
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 74
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 75
; SYM-NEXT:    Name: TIUninitIE
; SYM-NEXT:    Value (RelocatableAddress): 0x408
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 76
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 77
; SYM-NEXT:    Name: TIInitIE
; SYM-NEXT:    Value (RelocatableAddress): 0x410
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 78
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 79
; SYM-NEXT:    Name: TIUninitIE
; SYM-NEXT:    Value (RelocatableAddress): 0x418
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 80
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 81
; SYM-NEXT:    Name: TIInitLD
; SYM-NEXT:    Value (RelocatableAddress): 0x420
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 82
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 83
; SYM-NEXT:    Name: TIUninitLD
; SYM-NEXT:    Value (RelocatableAddress): 0x428
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 84
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 85
; SYM-NEXT:    Name: TIInitLD
; SYM-NEXT:    Value (RelocatableAddress): 0x430
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 86
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 87
; SYM-NEXT:    Name: TIUninitLD
; SYM-NEXT:    Value (RelocatableAddress): 0x438
; SYM-NEXT:    Section: .data
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 88
; SYM-NEXT:      SectionLen: 8
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 3
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TE (0x16)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 89
; SYM-NEXT:    Name: TIInitIE
; SYM-NEXT:    Value (RelocatableAddress): 0x0
; SYM-NEXT:    Section: .tdata
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 90
; SYM-NEXT:      SectionLen: 4
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 2
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 91
; SYM-NEXT:    Name: TIInitLD
; SYM-NEXT:    Value (RelocatableAddress): 0x4
; SYM-NEXT:    Section: .tdata
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 92
; SYM-NEXT:      SectionLen: 4
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 2
; SYM-NEXT:      SymbolType: XTY_SD (0x1)
; SYM-NEXT:      StorageMappingClass: XMC_TL (0x14)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 93
; SYM-NEXT:    Name: TIUninitIE
; SYM-NEXT:    Value (RelocatableAddress): 0x8
; SYM-NEXT:    Section: .tbss
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 94
; SYM-NEXT:      SectionLen: 4
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 2
; SYM-NEXT:      SymbolType: XTY_CM (0x3)
; SYM-NEXT:      StorageMappingClass: XMC_UL (0x15)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }
; SYM:    Index: 95
; SYM-NEXT:    Name: TIUninitLD
; SYM-NEXT:    Value (RelocatableAddress): 0xC
; SYM-NEXT:    Section: .tbss
; SYM-NEXT:    Type: 0x0
; SYM-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:    NumberOfAuxEntries: 1
; SYM-NEXT:    CSECT Auxiliary Entry {
; SYM-NEXT:      Index: 96
; SYM-NEXT:      SectionLen: 4
; SYM-NEXT:      ParameterHashIndex: 0x0
; SYM-NEXT:      TypeChkSectNum: 0x0
; SYM-NEXT:      SymbolAlignmentLog2: 2
; SYM-NEXT:      SymbolType: XTY_CM (0x3)
; SYM-NEXT:      StorageMappingClass: XMC_UL (0x15)
; SYM-NEXT:      Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:    }
; SYM-NEXT:  }

; DIS:      {{.*}}aix-func-use-tls-ld-ie-mix-reloc-large.ll.tmp.o:	file format aix5coff64-rs6000
; DIS:      Disassembly of section .data:
; DIS:      00000000000003f8 (idx: 71) _$TLSML[TC]:
; DIS-NEXT:     3f8: 00 00 00 00
; DIS-NEXT:	00000000000003f8:  R_TLSML	(idx: 71) _$TLSML[TC]
; DIS:      0000000000000400 (idx: 73) TIInitIE[TE]:
; DIS-NEXT:     400: 00 00 00 00
; DIS-NEXT:	0000000000000400:  R_TLS_LD	(idx: 89) TIInitIE[TL]
; DIS:      0000000000000408 (idx: 75) TIUninitIE[TE]:
; DIS-NEXT:     408: 00 00 00 00
; DIS-NEXT:	0000000000000408:  R_TLS_LD	(idx: 93) TIUninitIE[UL]
; DIS:      0000000000000410 (idx: 77) TIInitIE[TE]:
; DIS-NEXT:     410: 00 00 00 00
; DIS-NEXT:	0000000000000410:  R_TLS_IE	(idx: 89) TIInitIE[TL]
; DIS:      0000000000000418 (idx: 79) TIUninitIE[TE]:
; DIS-NEXT:     418: 00 00 00 00
; DIS-NEXT:	0000000000000418:  R_TLS_IE	(idx: 93) TIUninitIE[UL]
; DIS:      0000000000000420 (idx: 81) TIInitLD[TE]:
; DIS-NEXT:     420: 00 00 00 00
; DIS-NEXT:	0000000000000420:  R_TLS_IE	(idx: 91) TIInitLD[TL]
; DIS:      0000000000000428 (idx: 83) TIUninitLD[TE]:
; DIS-NEXT:     428: 00 00 00 00
; DIS-NEXT:	0000000000000428:  R_TLS_IE	(idx: 95) TIUninitLD[UL]
; DIS:      0000000000000430 (idx: 85) TIInitLD[TE]:
; DIS-NEXT:     430: 00 00 00 00
; DIS-NEXT:	0000000000000430:  R_TLS_LD	(idx: 91) TIInitLD[TL]
; DIS:      0000000000000438 (idx: 87) TIUninitLD[TE]:
; DIS-NEXT:     438: 00 00 00 00
; DIS-NEXT:	0000000000000438:  R_TLS_LD	(idx: 95) TIUninitLD[UL]

; DIS:      Disassembly of section .tdata:
; DIS:      0000000000000000 (idx: 89) TIInitIE[TL]:
; DIS-NEXT:       0: 00 00 00 2a
; DIS:      0000000000000004 (idx: 91) TIInitLD[TL]:
; DIS-NEXT:       4: 00 00 00 2a

; DIS:      Disassembly of section .tbss:
; DIS:      0000000000000008 (idx: 93) TIUninitIE[UL]:
; DIS-NEXT: ...
; DIS:      000000000000000c (idx: 95) TIUninitLD[UL]:
; DIS-NEXT: ...
