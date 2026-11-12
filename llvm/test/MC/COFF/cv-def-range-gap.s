# RUN: llvm-mc -triple=x86_64-pc-win32 -filetype=obj < %s | llvm-readobj --codeview - | FileCheck %s

# This tries to test defrange gap edge cases.

# CHECK:         LocalSym {
# CHECK:           Type: int (0x74)
# CHECK:           VarName: p
# CHECK:         }
# CHECK-NOT:     LocalSym {
# CHECK:         DefRangeRegisterSym {
# CHECK-NEXT:      Kind: S_DEFRANGE_REGISTER (0x1141)
# CHECK-NEXT:      Register: ESI (0x17)
# CHECK-NEXT:      MayHaveNoName: 0
# CHECK-NEXT:      LocalVariableAddrRange {
# CHECK-NEXT:        OffsetStart: .text+0x5
# CHECK-NEXT:        ISectStart: 0x0
# CHECK-NEXT:        Range: 0x5
# CHECK-NEXT:      }
# CHECK-NEXT:      LocalVariableAddrGap [
# CHECK-NEXT:        GapStartOffset: 0x3
# CHECK-NEXT:        Range: 0x1
# CHECK-NEXT:      ]
# CHECK-NEXT:    }
# CHECK-NEXT:    DefRangeRegisterSym {
# CHECK-NEXT:      Kind: S_DEFRANGE_REGISTER (0x1141)
# CHECK-NEXT:      Register: ESI (0x17)
# CHECK-NEXT:      MayHaveNoName: 0
# CHECK-NEXT:      LocalVariableAddrRange {
# CHECK-NEXT:        OffsetStart: .text+0x10015
# CHECK-NEXT:        ISectStart: 0x0
# CHECK-NEXT:        Range: 0x6
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:    DefRangeRegisterSym {
# CHECK-NEXT:      Kind: S_DEFRANGE_REGISTER (0x1141)
# CHECK-NEXT:      Register: ESI (0x17)
# CHECK-NEXT:      MayHaveNoName: 0
# CHECK-NEXT:      LocalVariableAddrRange {
# CHECK-NEXT:        OffsetStart: .text+0x2001B
# CHECK-NEXT:        ISectStart: 0x0
# CHECK-NEXT:        Range: 0x1
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:    DefRangeRegisterSym {
# CHECK-NEXT:      Kind: S_DEFRANGE_REGISTER (0x1141)
# CHECK-NEXT:      Register: ESI (0x17)
# CHECK-NEXT:      MayHaveNoName: 0
# CHECK-NEXT:      LocalVariableAddrRange {
# CHECK-NEXT:        OffsetStart: .text+0x2001C
# CHECK-NEXT:        ISectStart: 0x0
# CHECK-NEXT:        Range: 0xF000
# CHECK-NEXT:      }
# CHECK-NEXT:      LocalVariableAddrGap [
# CHECK-NEXT:        GapStartOffset: 0x1
# CHECK-NEXT:        Range: 0xEFFE
# CHECK-NEXT:      ]
# CHECK-NEXT:    }

	.text
f:                                      # @f
	mov $42, %esi
.Lbegin0:
	nop
	jmp .Lbegin1
.Lend0:
	nop
.Lbegin1:
	nop
.Lend1:
	.p2align	4
	.fill 0x10000, 1, 0x90

	mov $42, %esi
.Lbegin2:
	nop
	jmp .Lbegin3
.Lend2:
	.fill 0x10000, 1, 0x90
.Lbegin3:
	nop
.Lend3:

	# Create a range that is exactly 0xF000 bytes long with a gap in the
	# middle.
.Lbegin4:
	nop
.Lend4:
	.fill 0xeffe, 1, 0x90
.Lbegin5:
	nop
.Lend5:
	ret
.Lfunc_end0:

	.section	.debug$S,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	.long	241                     # Symbol subsection for f
	.long	.Ltmp15-.Ltmp14         # Subsection size
.Ltmp14:
	.short	.Ltmp17-.Ltmp16         # Record length
.Ltmp16:
	.short	4423                    # Record kind: S_GPROC32_ID
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	0                       # PtrNext
	.long	.Lfunc_end0-f           # Code size
	.long	0                       # Offset after prologue
	.long	0                       # Offset before epilogue
	.long	4098                    # Function type index
	.secrel32	f               # Function section relative address
	.secidx	f                       # Function section index
	.byte	0                       # Flags
	.asciz	"f"                     # Function name
.Ltmp17:
	.short	.Ltmp19-.Ltmp18         # Record length
.Ltmp18:
	.short	4414                    # Record kind: S_LOCAL
	.long	116                     # TypeIndex
	.short	0                       # Flags
	.asciz	"p"
.Ltmp19:
	.cv_def_range	 .Lbegin0 .Lend0 .Lbegin1 .Lend1 .Lbegin2 .Lend2 .Lbegin3 .Lend3, reg, 23
	.cv_def_range	 .Lbegin4 .Lend4 .Lbegin5 .Lend5, reg, 23
	.short	2                       # Record length
	.short	4431                    # Record kind: S_PROC_ID_END
.Ltmp15:
        .cv_filechecksums               # File index to string table offset subsection
        .cv_stringtable                 # String table
