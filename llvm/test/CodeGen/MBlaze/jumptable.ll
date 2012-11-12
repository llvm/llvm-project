; Ensure that jump tables can be handled by the mblaze backend. The
; jump table should be lowered to a "br" instruction using one of the
; available registers.
;
; RUN: llc < %s -march=mblaze | FileCheck %s

define i32 @jmptable(i32 %arg)
{
    ; CHECK:        jmptable:
    switch i32 %arg, label %DEFAULT [ i32 0, label %L0
                                      i32 1, label %L1
                                      i32 2, label %L2
                                      i32 3, label %L3
                                      i32 4, label %L4
                                      i32 5, label %L5
                                      i32 6, label %L6
                                      i32 7, label %L7
                                      i32 8, label %L8
                                      i32 9, label %L9 ]

    ; CHECK:        lw   [[REG:r[0-9]*]]
    ; CHECK:        brad [[REG]]
L0:
    %var0 = add i32 %arg, 0
    br label %DONE

L1:
    %var1 = add i32 %arg, 1
    br label %DONE

L2:
    %var2 = add i32 %arg, 2
    br label %DONE

L3:
    %var3 = add i32 %arg, 3
    br label %DONE

L4:
    %var4 = add i32 %arg, 4
    br label %DONE

L5:
    %var5 = add i32 %arg, 5
    br label %DONE

L6:
    %var6 = add i32 %arg, 6
    br label %DONE

L7:
    %var7 = add i32 %arg, 7
    br label %DONE

L8:
    %var8 = add i32 %arg, 8
    br label %DONE

L9:
    %var9 = add i32 %arg, 9
    br label %DONE

DEFAULT:
    unreachable

DONE:
    %rval = phi i32 [ %var0, %L0 ],
                    [ %var1, %L1 ],
                    [ %var2, %L2 ],
                    [ %var3, %L3 ],
                    [ %var4, %L4 ],
                    [ %var5, %L5 ],
                    [ %var6, %L6 ],
                    [ %var7, %L7 ],
                    [ %var8, %L8 ],
                    [ %var9, %L9 ]
    ret i32 %rval
    ; CHECK:        rtsd
}
