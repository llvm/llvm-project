; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck  -check-prefix=X64_DARWIN %s
; RUN: llc -mtriple=x86_64-pc-linux < %s | FileCheck  -check-prefix=X64_LINUX %s
; RUN: llc -mtriple=x86_64-pc-windows < %s | FileCheck  -check-prefix=X64_WINDOWS %s
; RUN: llc -mtriple=x86_64-pc-windows-gnu < %s | FileCheck  -check-prefix=X64_WINDOWS_GNU %s
; RUN: llc -mtriple=x86_64-scei-ps4 < %s | FileCheck -check-prefix=PS4 %s

; X64_DARWIN: orq
; X64_DARWIN-NEXT: ud2

; X64_LINUX: orq _ZN11xercesc_2_56XMLUni16fgNotationStringE@GOTPCREL(%rip), %rax
; X64_LINUX-NEXT: jne
; X64_LINUX-NEXT: %bb8.i329

; X64_WINDOWS: orq %rax, %rcx
; X64_WINDOWS-NEXT: jne

; X64_WINDOWS_GNU: movq .refptr._ZN11xercesc_2_513SchemaSymbols21fgURI_SCHEMAFORSCHEMAE(%rip), %rax
; X64_WINDOWS_GNU: orq .refptr._ZN11xercesc_2_56XMLUni16fgNotationStringE(%rip), %rax
; X64_WINDOWS_GNU-NEXT: jne

; PS4: orq _ZN11xercesc_2_56XMLUni16fgNotationStringE@GOTPCREL(%rip), %rax
; PS4-NEXT: ud2

@_ZN11xercesc_2_513SchemaSymbols21fgURI_SCHEMAFORSCHEMAE = external constant [33 x i16], align 32 ; <ptr> [#uses=1]
@_ZN11xercesc_2_56XMLUni16fgNotationStringE = external constant [9 x i16], align 16 ; <ptr> [#uses=1]

define fastcc void @foo() {
entry:
  br i1 icmp eq (i64 or (i64 ptrtoint (ptr @_ZN11xercesc_2_513SchemaSymbols21fgURI_SCHEMAFORSCHEMAE to i64),
                         i64 ptrtoint (ptr @_ZN11xercesc_2_56XMLUni16fgNotationStringE to i64)), i64 0),
     label %bb8.i329, label %bb4.i.i318.preheader

bb4.i.i318.preheader:                             ; preds = %bb6
  unreachable

bb8.i329:                                         ; preds = %bb6
  unreachable
}
