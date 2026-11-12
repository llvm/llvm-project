; RUN: llc < %s -O0

define float @test(i32 %tmp12771278) {
        switch i32 %tmp12771278, label %bb1279 [
        ]

bb1279:         ; preds = %0
        ret float 1.000000e+00
}

