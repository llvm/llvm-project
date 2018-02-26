; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64--

        %struct.TCMalloc_SpinLock = type { i32 }

define void @_ZN17TCMalloc_SpinLock4LockEv(%struct.TCMalloc_SpinLock* %this) {
entry:
        %tmp3 = call i32 asm sideeffect "1: lwarx $0, 0, $1\0A\09stwcx. $2, 0, $1\0A\09bne- 1b\0A\09isync", "=&r,=*r,r,1,~{dirflag},~{fpsr},~{flags},~{memory}"( i32** null, i32 1, i32* null )         ; <i32> [#uses=0]
        unreachable
}
