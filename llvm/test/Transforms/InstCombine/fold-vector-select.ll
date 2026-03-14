; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; CHECK-NOT: select

define void @foo(ptr %A, ptr %B, ptr %C, ptr %D,
                 ptr %E, ptr %F, ptr %G, ptr %H,
                 ptr %I, ptr %J, ptr %K, ptr %L,
                 ptr %M, ptr %N, ptr %O, ptr %P,
                 ptr %Q, ptr %R, ptr %S, ptr %T,
                 ptr %U, ptr %V, ptr %W, ptr %X,
                 ptr %Y, ptr %Z, ptr %BA, ptr %BB,
                 ptr %BC, ptr %BD, ptr %BE, ptr %BF,
                 ptr %BG, ptr %BH, ptr %BI, ptr %BJ,
                 ptr %BK, ptr %BL, ptr %BM, ptr %BN,
                 ptr %BO, ptr %BP, ptr %BQ, ptr %BR,
                 ptr %BS, ptr %BT, ptr %BU, ptr %BV,
                 ptr %BW, ptr %BX, ptr %BY, ptr %BZ,
                 ptr %CA, ptr %CB, ptr %CC, ptr %CD,
                 ptr %CE, ptr %CF, ptr %CG, ptr %CH,
                 ptr %CI, ptr %CJ, ptr %CK, ptr %CL) {
 %a = select <4 x i1> <i1 false, i1 false, i1 false, i1 false>, <4 x i32> zeroinitializer, <4 x i32> <i32 9, i32 87, i32 57, i32 8>
 %b = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i32> zeroinitializer, <4 x i32> <i32 44, i32 99, i32 49, i32 29>
 %c = select <4 x i1> <i1 false, i1 true, i1 false, i1 false>, <4 x i32> zeroinitializer, <4 x i32> <i32 15, i32 18, i32 53, i32 84>
 %d = select <4 x i1> <i1 true, i1 true, i1 false, i1 false>, <4 x i32> zeroinitializer, <4 x i32> <i32 29, i32 82, i32 45, i32 16>
 %e = select <4 x i1> <i1 false, i1 false, i1 true, i1 false>, <4 x i32> zeroinitializer, <4 x i32> <i32 11, i32 15, i32 32, i32 99>
 %f = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i32> zeroinitializer, <4 x i32> <i32 19, i32 86, i32 29, i32 33>
 %g = select <4 x i1> <i1 false, i1 true, i1 true, i1 false>, <4 x i32> zeroinitializer, <4 x i32> <i32 44, i32 10, i32 26, i32 45>
 %h = select <4 x i1> <i1 true, i1 true, i1 true, i1 false>, <4 x i32> zeroinitializer, <4 x i32> <i32 88, i32 70, i32 90, i32 48>
 %i = select <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x i32> zeroinitializer, <4 x i32> <i32 30, i32 53, i32 42, i32 12>
 %j = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x i32> zeroinitializer, <4 x i32> <i32 46, i32 24, i32 93, i32 26>
 %k = select <4 x i1> <i1 false, i1 true, i1 false, i1 true>, <4 x i32> zeroinitializer, <4 x i32> <i32 33, i32 99, i32 15, i32 57>
 %l = select <4 x i1> <i1 true, i1 true, i1 false, i1 true>, <4 x i32> zeroinitializer, <4 x i32> <i32 51, i32 60, i32 60, i32 50>
 %m = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x i32> zeroinitializer, <4 x i32> <i32 50, i32 12, i32 7, i32 45>
 %n = select <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x i32> zeroinitializer, <4 x i32> <i32 15, i32 65, i32 36, i32 36>
 %o = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x i32> zeroinitializer, <4 x i32> <i32 54, i32 0, i32 17, i32 78>
 %p = select <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> zeroinitializer, <4 x i32> <i32 56, i32 13, i32 64, i32 48>
 %q = select <4 x i1> <i1 false, i1 false, i1 false, i1 false>, <4 x i32> <i32 52, i32 69, i32 88, i32 11>, <4 x i32> zeroinitializer
 %r = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i32> <i32 5, i32 87, i32 68, i32 14>, <4 x i32> zeroinitializer
 %s = select <4 x i1> <i1 false, i1 true, i1 false, i1 false>, <4 x i32> <i32 47, i32 17, i32 66, i32 63>, <4 x i32> zeroinitializer
 %t = select <4 x i1> <i1 true, i1 true, i1 false, i1 false>, <4 x i32> <i32 64, i32 25, i32 73, i32 81>, <4 x i32> zeroinitializer
 %u = select <4 x i1> <i1 false, i1 false, i1 true, i1 false>, <4 x i32> <i32 51, i32 41, i32 61, i32 63>, <4 x i32> zeroinitializer
 %v = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i32> <i32 39, i32 59, i32 17, i32 0>, <4 x i32> zeroinitializer
 %w = select <4 x i1> <i1 false, i1 true, i1 true, i1 false>, <4 x i32> <i32 91, i32 99, i32 97, i32 29>, <4 x i32> zeroinitializer
 %x = select <4 x i1> <i1 true, i1 true, i1 true, i1 false>, <4 x i32> <i32 89, i32 45, i32 89, i32 10>, <4 x i32> zeroinitializer
 %y = select <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x i32> <i32 25, i32 70, i32 21, i32 27>, <4 x i32> zeroinitializer
 %z = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x i32> <i32 40, i32 12, i32 27, i32 88>, <4 x i32> zeroinitializer
 %ba = select <4 x i1> <i1 false, i1 true, i1 false, i1 true>, <4 x i32> <i32 36, i32 35, i32 90, i32 23>, <4 x i32> zeroinitializer
 %bb = select <4 x i1> <i1 true, i1 true, i1 false, i1 true>, <4 x i32> <i32 83, i32 3, i32 64, i32 82>, <4 x i32> zeroinitializer
 %bc = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x i32> <i32 15, i32 72, i32 2, i32 54>, <4 x i32> zeroinitializer
 %bd = select <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x i32> <i32 32, i32 47, i32 100, i32 84>, <4 x i32> zeroinitializer
 %be = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x i32> <i32 92, i32 57, i32 82, i32 1>, <4 x i32> zeroinitializer
 %bf = select <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> <i32 42, i32 14, i32 22, i32 89>, <4 x i32> zeroinitializer
 %bg = select <4 x i1> <i1 false, i1 false, i1 false, i1 false>, <4 x i32> <i32 33, i32 10, i32 67, i32 66>, <4 x i32> <i32 42, i32 91, i32 47, i32 40>
 %bh = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i32> <i32 8, i32 13, i32 48, i32 0>, <4 x i32> <i32 84, i32 66, i32 87, i32 84>
 %bi = select <4 x i1> <i1 false, i1 true, i1 false, i1 false>, <4 x i32> <i32 85, i32 96, i32 1, i32 94>, <4 x i32> <i32 54, i32 57, i32 7, i32 92>
 %bj = select <4 x i1> <i1 true, i1 true, i1 false, i1 false>, <4 x i32> <i32 55, i32 21, i32 92, i32 68>, <4 x i32> <i32 51, i32 61, i32 62, i32 39>
 %bk = select <4 x i1> <i1 false, i1 false, i1 true, i1 false>, <4 x i32> <i32 42, i32 18, i32 77, i32 74>, <4 x i32> <i32 82, i32 33, i32 30, i32 7>
 %bl = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i32> <i32 80, i32 92, i32 61, i32 84>, <4 x i32> <i32 43, i32 89, i32 92, i32 6>
 %bm = select <4 x i1> <i1 false, i1 true, i1 true, i1 false>, <4 x i32> <i32 49, i32 14, i32 62, i32 62>, <4 x i32> <i32 35, i32 33, i32 92, i32 59>
 %bn = select <4 x i1> <i1 true, i1 true, i1 true, i1 false>, <4 x i32> <i32 3, i32 97, i32 49, i32 18>, <4 x i32> <i32 56, i32 64, i32 19, i32 75>
 %bo = select <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x i32> <i32 91, i32 57, i32 0, i32 1>, <4 x i32> <i32 43, i32 63, i32 64, i32 11>
 %bp = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x i32> <i32 41, i32 65, i32 18, i32 11>, <4 x i32> <i32 86, i32 26, i32 31, i32 3>
 %bq = select <4 x i1> <i1 false, i1 true, i1 false, i1 true>, <4 x i32> <i32 31, i32 46, i32 32, i32 68>, <4 x i32> <i32 100, i32 59, i32 62, i32 6>
 %br = select <4 x i1> <i1 true, i1 true, i1 false, i1 true>, <4 x i32> <i32 76, i32 67, i32 87, i32 7>, <4 x i32> <i32 63, i32 48, i32 97, i32 24>
 %bs = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x i32> <i32 83, i32 89, i32 19, i32 4>, <4 x i32> <i32 21, i32 2, i32 40, i32 21>
 %bt = select <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x i32> <i32 45, i32 76, i32 81, i32 100>, <4 x i32> <i32 65, i32 26, i32 100, i32 46>
 %bu = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x i32> <i32 16, i32 75, i32 31, i32 17>, <4 x i32> <i32 37, i32 66, i32 86, i32 65>
 %bv = select <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> <i32 13, i32 25, i32 43, i32 59>, <4 x i32> <i32 82, i32 78, i32 60, i32 52>
 %bw = select <4 x i1> <i1 false, i1 false, i1 false, i1 false>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %bx = select <4 x i1> <i1 true, i1 false, i1 false, i1 false>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %by = select <4 x i1> <i1 false, i1 true, i1 false, i1 false>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %bz = select <4 x i1> <i1 true, i1 true, i1 false, i1 false>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %ca = select <4 x i1> <i1 false, i1 false, i1 true, i1 false>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %cb = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %cc = select <4 x i1> <i1 false, i1 true, i1 true, i1 false>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %cd = select <4 x i1> <i1 true, i1 true, i1 true, i1 false>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %ce = select <4 x i1> <i1 false, i1 false, i1 false, i1 true>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %cf = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %cg = select <4 x i1> <i1 false, i1 true, i1 false, i1 true>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %ch = select <4 x i1> <i1 true, i1 true, i1 false, i1 true>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %ci = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %cj = select <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %ck = select <4 x i1> <i1 false, i1 true, i1 true, i1 true>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 %cl = select <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer
 store <4 x i32> %a, ptr %A
 store <4 x i32> %b, ptr %B
 store <4 x i32> %c, ptr %C
 store <4 x i32> %d, ptr %D
 store <4 x i32> %e, ptr %E
 store <4 x i32> %f, ptr %F
 store <4 x i32> %g, ptr %G
 store <4 x i32> %h, ptr %H
 store <4 x i32> %i, ptr %I
 store <4 x i32> %j, ptr %J
 store <4 x i32> %k, ptr %K
 store <4 x i32> %l, ptr %L
 store <4 x i32> %m, ptr %M
 store <4 x i32> %n, ptr %N
 store <4 x i32> %o, ptr %O
 store <4 x i32> %p, ptr %P
 store <4 x i32> %q, ptr %Q
 store <4 x i32> %r, ptr %R
 store <4 x i32> %s, ptr %S
 store <4 x i32> %t, ptr %T
 store <4 x i32> %u, ptr %U
 store <4 x i32> %v, ptr %V
 store <4 x i32> %w, ptr %W
 store <4 x i32> %x, ptr %X
 store <4 x i32> %y, ptr %Y
 store <4 x i32> %z, ptr %Z
 store <4 x i32> %ba, ptr %BA
 store <4 x i32> %bb, ptr %BB
 store <4 x i32> %bc, ptr %BC
 store <4 x i32> %bd, ptr %BD
 store <4 x i32> %be, ptr %BE
 store <4 x i32> %bf, ptr %BF
 store <4 x i32> %bg, ptr %BG
 store <4 x i32> %bh, ptr %BH
 store <4 x i32> %bi, ptr %BI
 store <4 x i32> %bj, ptr %BJ
 store <4 x i32> %bk, ptr %BK
 store <4 x i32> %bl, ptr %BL
 store <4 x i32> %bm, ptr %BM
 store <4 x i32> %bn, ptr %BN
 store <4 x i32> %bo, ptr %BO
 store <4 x i32> %bp, ptr %BP
 store <4 x i32> %bq, ptr %BQ
 store <4 x i32> %br, ptr %BR
 store <4 x i32> %bs, ptr %BS
 store <4 x i32> %bt, ptr %BT
 store <4 x i32> %bu, ptr %BU
 store <4 x i32> %bv, ptr %BV
 store <4 x i32> %bw, ptr %BW
 store <4 x i32> %bx, ptr %BX
 store <4 x i32> %by, ptr %BY
 store <4 x i32> %bz, ptr %BZ
 store <4 x i32> %ca, ptr %CA
 store <4 x i32> %cb, ptr %CB
 store <4 x i32> %cc, ptr %CC
 store <4 x i32> %cd, ptr %CD
 store <4 x i32> %ce, ptr %CE
 store <4 x i32> %cf, ptr %CF
 store <4 x i32> %cg, ptr %CG
 store <4 x i32> %ch, ptr %CH
 store <4 x i32> %ci, ptr %CI
 store <4 x i32> %cj, ptr %CJ
 store <4 x i32> %ck, ptr %CK
 store <4 x i32> %cl, ptr %CL
 ret void
}
