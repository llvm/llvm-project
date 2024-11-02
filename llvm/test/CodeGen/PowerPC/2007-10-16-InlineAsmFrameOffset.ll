; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--
; rdar://5538377

        %struct.disk_unsigned = type { i32 }
        %struct._StorePageMax = type { %struct.disk_unsigned, %struct.disk_unsigned, [65536 x i8] }

define i32 @test() {
entry:
        %data = alloca i32              ; <ptr> [#uses=1]
        %compressedPage = alloca %struct._StorePageMax          ; <ptr> [#uses=0]
        %tmp107 = call i32 asm "lwbrx $0, $2, $1", "=r,r,bO,*m"( ptr null, i32 0, ptr elementtype(i32) %data )          ; <i32> [#uses=0]
        unreachable
}

