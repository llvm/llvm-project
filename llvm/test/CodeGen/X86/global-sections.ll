; RUN: llc < %s -mtriple=i386-unknown-linux-gnu | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=i386-apple-darwin9.7 | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=i386-apple-darwin10 -relocation-model=static | FileCheck %s -check-prefix=DARWIN-STATIC
; RUN: llc < %s -mtriple=x86_64-apple-darwin10 | FileCheck %s -check-prefix=DARWIN64
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu -data-sections -function-sections | FileCheck %s -check-prefix=LINUX-SECTIONS
; RUN: llc < %s -mtriple=i386-unknown-linux-gnu -function-sections | FileCheck %s -check-prefix=LINUX-FUNC-SECTIONS
; RUN: llc < %s -mtriple=x86_64-pc-linux -data-sections -function-sections -relocation-model=pic | FileCheck %s -check-prefix=LINUX-SECTIONS-PIC
; RUN: llc < %s -mtriple=i686-pc-win32 -data-sections -function-sections | FileCheck %s -check-prefix=WIN32-SECTIONS
; RUN: llc < %s -mtriple=i686-pc-win32 -function-sections | FileCheck %s -check-prefix=WIN32-FUNC-SECTIONS

define void @F1() {
  ret void
}

; WIN32-SECTIONS: .section        .text,"xr",one_only,_F1
; WIN32-SECTIONS: .globl _F1

define void @F2(i32 %y) {
bb0:
switch i32 %y, label %bb5 [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
  ]
bb1:
  ret void
bb2:
  ret void
bb3:
  ret void
bb4:
  ret void
bb5:
  ret void
}

; LINUX:     .size   F2,
; LINUX-NEX: .cfi_endproc
; LINUX-NEX: .section        .rodata,"a",@progbits

; LINUX-SECTIONS: .section        .text.F2,"ax",@progbits
; LINUX-SECTIONS: .size   F2,
; LINUX-SECTIONS-NEXT: .cfi_endproc
; LINUX-SECTIONS-NEXT: .section        .rodata.F2,"a",@progbits

; LINUX-FUNC-SECTIONS: .section        .text.F2,"ax",@progbits
; LINUX-FUNC-SECTIONS: .size   F2,
; LINUX-FUNC-SECTIONS-NEXT: .cfi_endproc
; LINUX-FUNC-SECTIONS-NEXT: .section        .rodata.F2,"a",@progbits

; WIN32-FUNC-SECTIONS: .section        .text,"xr",one_only,_F2
; WIN32-FUNC-SECTIONS-NOT: .section
; WIN32-FUNC-SECTIONS: .section        .rdata,"dr",associative,_F2


; LINUX-SECTIONS-PIC: .section        .text.F2,"ax",@progbits
; LINUX-SECTIONS-PIC: .size   F2,
; LINUX-SECTIONS-PIC-NEXT: .cfi_endproc
; LINUX-SECTIONS-PIC-NEXT: .section        .rodata.F2,"a",@progbits

declare void @G()

define void @F3(i32 %y) personality i8* bitcast (void ()* @G to i8*) {
bb0:
  invoke void @G()
          to label %bb2 unwind label %bb1
bb1:
  landingpad { i8*, i32 }
          catch i8* null
  br label %bb2
bb2:

switch i32 %y, label %bb7 [
    i32 1, label %bb3
    i32 2, label %bb4
    i32 3, label %bb5
    i32 4, label %bb6
  ]
bb3:
  ret void
bb4:
  ret void
bb5:
  ret void
bb6:
  ret void
bb7:
  ret void
}

; DARWIN64: _F3:
; DARWIN64: Lfunc_end
; DARWIN64-NEXT: .cfi_endproc
; DARWIN64-NOT: .section
; DARWIN64: .data_region jt32
; DARWIN64: LJTI{{.*}}:
; DARWIN64-NEXT: .long
; DARWIN64-NEXT: .long
; DARWIN64-NEXT: .long
; DARWIN64-NEXT: .long
; DARWIN64-NEXT: .end_data_region
; DARWIN64-NEXT: .section        __TEXT,__gcc_except_tab

; int G1;
@G1 = common global i32 0

; LINUX: .type   G1,@object
; LINUX: .comm  G1,4,4

; DARWIN: .comm _G1,4,2




; const int G2 __attribute__((weak)) = 42;
@G2 = weak_odr unnamed_addr constant i32 42     


; TODO: linux drops this into .rodata, we drop it into ".gnu.linkonce.r.G2"

; DARWIN: .section __TEXT,__const{{$}}
; DARWIN: _G2:
; DARWIN:    .long 42


; int * const G3 = &G1;
@G3 = unnamed_addr constant i32* @G1

; DARWIN: .section        __DATA,__const
; DARWIN: .globl _G3
; DARWIN: _G3:
; DARWIN:     .long _G1

; LINUX:   .section        .rodata,"a",@progbits
; LINUX:   .globl  G3

; LINUX-SECTIONS: .section        .rodata.G3,"a",@progbits
; LINUX-SECTIONS: .globl  G3

; WIN32-SECTIONS: .section        .rdata,"dr",one_only,_G3
; WIN32-SECTIONS: .globl  _G3


; _Complex long long const G4 = 34;
@G4 = private unnamed_addr constant {i64,i64} { i64 34, i64 0 }

; DARWIN: .section        __TEXT,__literal16,16byte_literals
; DARWIN: L_G4:
; DARWIN:     .long 34

; DARWIN-STATIC: .section        __TEXT,__literal16,16byte_literals
; DARWIN-STATIC: L_G4:
; DARWIN-STATIC:     .long 34

; DARWIN64: .section        __TEXT,__literal16,16byte_literals
; DARWIN64: L_G4:
; DARWIN64:     .quad 34

; int G5 = 47;
@G5 = global i32 47

; LINUX: .data
; LINUX: .globl G5
; LINUX: G5:
; LINUX:    .long 47

; DARWIN: .section        __DATA,__data
; DARWIN: .globl _G5
; DARWIN: _G5:
; DARWIN:    .long 47


; PR4584
@"foo bar" = linkonce global i32 42

; LINUX: .type  "foo bar",@object
; LINUX: .weak  "foo bar"
; LINUX: "foo bar":

; DARWIN: .globl        "_foo bar"
; DARWIN:       .weak_definition "_foo bar"
; DARWIN: "_foo bar":

; PR4650
@G6 = weak_odr unnamed_addr constant [1 x i8] c"\01"

; LINUX:   .type        G6,@object
; LINUX:   .weak        G6
; LINUX: G6:
; LINUX:   .byte        1
; LINUX:   .size        G6, 1

; DARWIN:  .section __TEXT,__const{{$}}
; DARWIN:  .globl _G6
; DARWIN:  .weak_definition _G6
; DARWIN:_G6:
; DARWIN:  .byte 1


@G7 = unnamed_addr constant [10 x i8] c"abcdefghi\00"

; DARWIN:       __TEXT,__cstring,cstring_literals
; DARWIN:       .globl _G7
; DARWIN: _G7:
; DARWIN:       .asciz  "abcdefghi"

; LINUX:        .section        .rodata.str1.1,"aMS",@progbits,1
; LINUX:        .globl G7
; LINUX: G7:
; LINUX:        .asciz  "abcdefghi"

; LINUX-SECTIONS: .section        .rodata.str1.1,"aMS",@progbits,1
; LINUX-SECTIONS:       .globl G7

; WIN32-SECTIONS: .section        .rdata,"dr",one_only,_G7
; WIN32-SECTIONS:       .globl _G7


@G8 = unnamed_addr constant [4 x i16] [ i16 1, i16 2, i16 3, i16 0 ]

; DARWIN:       .section        __TEXT,__const
; DARWIN:       .globl _G8
; DARWIN: _G8:

; LINUX:        .section        .rodata.str2.2,"aMS",@progbits,2
; LINUX:        .globl G8
; LINUX:G8:

@G9 = unnamed_addr constant [4 x i32] [ i32 1, i32 2, i32 3, i32 0 ]

; DARWIN:       .globl _G9
; DARWIN: _G9:

; LINUX:        .section        .rodata.str4.4,"aMS",@progbits,4
; LINUX:        .globl G9
; LINUX:G9


@G10 = weak global [100 x i32] zeroinitializer, align 32 ; <[100 x i32]*> [#uses=0]


; DARWIN:       .section        __DATA,__data{{$}}
; DARWIN: .globl _G10
; DARWIN:       .weak_definition _G10
; DARWIN:       .p2align  5
; DARWIN: _G10:
; DARWIN:       .space  400

; LINUX:        .bss
; LINUX:        .weak   G10
; LINUX:        .p2align  5
; LINUX: G10:
; LINUX:        .zero   400



;; Zero sized objects should round up to 1 byte in zerofill directives.
; rdar://7886017
@G11 = global [0 x i32] zeroinitializer
@G12 = global {} zeroinitializer
@G13 = global { [0 x {}] } zeroinitializer

; DARWIN: .globl _G11
; DARWIN: .zerofill __DATA,__common,_G11,1,2
; DARWIN: .globl _G12
; DARWIN: .zerofill __DATA,__common,_G12,1,3
; DARWIN: .globl _G13
; DARWIN: .zerofill __DATA,__common,_G13,1,3

@G14 = private unnamed_addr constant [4 x i8] c"foo\00", align 1

; LINUX-SECTIONS:        .type   .LG14,@object           # @G14
; LINUX-SECTIONS:        .section        .rodata.str1.1,"aMS",@progbits,1
; LINUX-SECTIONS: .LG14:
; LINUX-SECTIONS:        .asciz  "foo"
; LINUX-SECTIONS:        .size   .LG14, 4

; WIN32-SECTIONS:        .section        .rdata,"dr",one_only,_G14
; WIN32-SECTIONS: _G14:
; WIN32-SECTIONS:        .asciz  "foo"

; cannot be merged on MachO, but can on other formats.
@G15 = unnamed_addr constant i64 0

; LINUX: .section        .rodata.cst8,"aM",@progbits,8
; LINUX: G15:

; DARWIN: .section      __TEXT,__const
; DARWIN: _G15:

; DARWIN-STATIC: .section       __TEXT,__const
; DARWIN-STATIC: _G15:

; DARWIN64: .section       __TEXT,__const
; DARWIN64: _G15:

; LINUX-SECTIONS: .section      .rodata.cst8,"aM",@progbits,8
; LINUX-SECTIONS: G15:

; WIN32-SECTIONS: .section      .rdata,"dr",one_only,_G15
; WIN32-SECTIONS: _G15:

@G16 = unnamed_addr constant i256 0

; LINUX: .section        .rodata.cst32,"aM",@progbits,32
; LINUX: G16:

; LINUX-SECTIONS: .section      .rodata.cst32,"aM",@progbits,32
; LINUX-SECTIONS: G16:

; WIN32-SECTIONS: .section      .rdata,"dr",one_only,_G16
; WIN32-SECTIONS: _G16:

; PR26570

@G17 = internal global i8 0
; LINUX: .type	G17,@object
; LINUX: .local	G17
; LINUX: .comm	G17,1,1

; DARWIN: .zerofill __DATA,__bss,_G17,1,0

; LINUX-SECTIONS: .type	G17,@object
; LINUX-SECTIONS: .section	.bss.G17,"aw",@nobits
; LINUX-SECTIONS: .byte	0
; LINUX-SECTIONS: .size	G17, 1

; WIN32-SECTIONS: .section	.bss,"bw",one_only,_G17
; WIN32-SECTIONS: _G17:
; WIN32-SECTIONS:.byte	0
