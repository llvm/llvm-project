; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep rlwimi

define void @test(i16 %div.0.i.i.i.i, i32 %L_num.0.i.i.i.i, i32 %tmp1.i.i206.i.i, i16* %P) {
        %X = shl i16 %div.0.i.i.i.i, 1          ; <i16> [#uses=1]
        %tmp28.i.i.i.i = shl i32 %L_num.0.i.i.i.i, 1            ; <i32> [#uses=1]
        %tmp31.i.i.i.i = icmp slt i32 %tmp28.i.i.i.i, %tmp1.i.i206.i.i          ; <i1> [#uses=1]
        %tmp31.i.i.i.i.upgrd.1 = zext i1 %tmp31.i.i.i.i to i16          ; <i16> [#uses=1]
        %tmp371.i.i.i.i1 = or i16 %tmp31.i.i.i.i.upgrd.1, %X            ; <i16> [#uses=1]
        %div.0.be.i.i.i.i = xor i16 %tmp371.i.i.i.i1, 1         ; <i16> [#uses=1]
        store i16 %div.0.be.i.i.i.i, i16* %P
        ret void
}

