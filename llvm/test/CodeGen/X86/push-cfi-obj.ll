; RUN: llc < %s -mtriple=i686-pc-linux -filetype=obj | llvm-readobj -S --sr --sd | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=i686-darwin-macosx10.7 -filetype=obj | llvm-readobj --sections | FileCheck -check-prefix=DARWIN %s

; On darwin, check that we manage to generate the compact unwind section
; DARWIN: Name: __compact_unwind
; DARWIN: Segment: __LD

; LINUX:         Name: .eh_frame
; LINUX-NEXT:    Type: SHT_PROGBITS (0x1)
; LINUX-NEXT:    Flags [ (0x2)
; LINUX-NEXT:      SHF_ALLOC (0x2)
; LINUX-NEXT:    ]
; LINUX-NEXT:    Address: 0x0
; LINUX-NEXT:    Offset: 0x5C
; LINUX-NEXT:    Size: 72
; LINUX-NEXT:    Link: 0
; LINUX-NEXT:    Info: 0
; LINUX-NEXT:    AddressAlignment: 4
; LINUX-NEXT:    EntrySize: 0
; LINUX-NEXT:    Relocations [
; LINUX-NEXT:    ]
; LINUX-NEXT:    SectionData (
; LINUX-NEXT:      0000: 1C000000 00000000 017A504C 5200017C  |.........zPLR..||
; LINUX-NEXT:      0010: 08070000 00000000 1B0C0404 88010000  |................|
; LINUX-NEXT:      0020: 24000000 24000000 00000000 1D000000  |$...$...........|
; LINUX-NEXT:      0030: 04000000 00410E08 8502420D 05432E10  |.....A....B..C..|
; LINUX-NEXT:      0040: 540C0404 410C0508                    |T...A...|
; LINUX-NEXT:    )

declare i32 @__gxx_personality_v0(...)
declare void @good(i32 %a, i32 %b, i32 %c, i32 %d)

define void @test() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @good(i32 1, i32 2, i32 3, i32 4)
          to label %continue unwind label %cleanup
continue:
  ret void
cleanup:
  landingpad { i8*, i32 }
     cleanup
  ret void
}

attributes #0 = { optsize "no-frame-pointer-elim"="true" }
