; RUN: opt < %s -instcombine -S | not grep select

define void @foo(<4 x i32> *%A, <4 x i32> *%B, <4 x i32> *%C, <4 x i32> *%D,
                 <4 x i32> *%E, <4 x i32> *%F, <4 x i32> *%G, <4 x i32> *%H,
                 <4 x i32> *%I, <4 x i32> *%J, <4 x i32> *%K, <4 x i32> *%L,
                 <4 x i32> *%M, <4 x i32> *%N, <4 x i32> *%O, <4 x i32> *%P,
                 <4 x i32> *%Q, <4 x i32> *%R, <4 x i32> *%S, <4 x i32> *%T,
                 <4 x i32> *%U, <4 x i32> *%V, <4 x i32> *%W, <4 x i32> *%X,
                 <4 x i32> *%Y, <4 x i32> *%Z, <4 x i32> *%BA, <4 x i32> *%BB,
                 <4 x i32> *%BC, <4 x i32> *%BD, <4 x i32> *%BE, <4 x i32> *%BF,
                 <4 x i32> *%BG, <4 x i32> *%BH, <4 x i32> *%BI, <4 x i32> *%BJ,
                 <4 x i32> *%BK, <4 x i32> *%BL, <4 x i32> *%BM, <4 x i32> *%BN,
                 <4 x i32> *%BO, <4 x i32> *%BP, <4 x i32> *%BQ, <4 x i32> *%BR,
                 <4 x i32> *%BS, <4 x i32> *%BT, <4 x i32> *%BU, <4 x i32> *%BV,
                 <4 x i32> *%BW, <4 x i32> *%BX, <4 x i32> *%BY, <4 x i32> *%BZ,
                 <4 x i32> *%CA, <4 x i32> *%CB, <4 x i32> *%CC, <4 x i32> *%CD,
                 <4 x i32> *%CE, <4 x i32> *%CF, <4 x i32> *%CG, <4 x i32> *%CH,
                 <4 x i32> *%CI, <4 x i32> *%CJ, <4 x i32> *%CK, <4 x i32> *%CL) {
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
 store <4 x i32> %a, <4 x i32>* %A
 store <4 x i32> %b, <4 x i32>* %B
 store <4 x i32> %c, <4 x i32>* %C
 store <4 x i32> %d, <4 x i32>* %D
 store <4 x i32> %e, <4 x i32>* %E
 store <4 x i32> %f, <4 x i32>* %F
 store <4 x i32> %g, <4 x i32>* %G
 store <4 x i32> %h, <4 x i32>* %H
 store <4 x i32> %i, <4 x i32>* %I
 store <4 x i32> %j, <4 x i32>* %J
 store <4 x i32> %k, <4 x i32>* %K
 store <4 x i32> %l, <4 x i32>* %L
 store <4 x i32> %m, <4 x i32>* %M
 store <4 x i32> %n, <4 x i32>* %N
 store <4 x i32> %o, <4 x i32>* %O
 store <4 x i32> %p, <4 x i32>* %P
 store <4 x i32> %q, <4 x i32>* %Q
 store <4 x i32> %r, <4 x i32>* %R
 store <4 x i32> %s, <4 x i32>* %S
 store <4 x i32> %t, <4 x i32>* %T
 store <4 x i32> %u, <4 x i32>* %U
 store <4 x i32> %v, <4 x i32>* %V
 store <4 x i32> %w, <4 x i32>* %W
 store <4 x i32> %x, <4 x i32>* %X
 store <4 x i32> %y, <4 x i32>* %Y
 store <4 x i32> %z, <4 x i32>* %Z
 store <4 x i32> %ba, <4 x i32>* %BA
 store <4 x i32> %bb, <4 x i32>* %BB
 store <4 x i32> %bc, <4 x i32>* %BC
 store <4 x i32> %bd, <4 x i32>* %BD
 store <4 x i32> %be, <4 x i32>* %BE
 store <4 x i32> %bf, <4 x i32>* %BF
 store <4 x i32> %bg, <4 x i32>* %BG
 store <4 x i32> %bh, <4 x i32>* %BH
 store <4 x i32> %bi, <4 x i32>* %BI
 store <4 x i32> %bj, <4 x i32>* %BJ
 store <4 x i32> %bk, <4 x i32>* %BK
 store <4 x i32> %bl, <4 x i32>* %BL
 store <4 x i32> %bm, <4 x i32>* %BM
 store <4 x i32> %bn, <4 x i32>* %BN
 store <4 x i32> %bo, <4 x i32>* %BO
 store <4 x i32> %bp, <4 x i32>* %BP
 store <4 x i32> %bq, <4 x i32>* %BQ
 store <4 x i32> %br, <4 x i32>* %BR
 store <4 x i32> %bs, <4 x i32>* %BS
 store <4 x i32> %bt, <4 x i32>* %BT
 store <4 x i32> %bu, <4 x i32>* %BU
 store <4 x i32> %bv, <4 x i32>* %BV
 store <4 x i32> %bw, <4 x i32>* %BW
 store <4 x i32> %bx, <4 x i32>* %BX
 store <4 x i32> %by, <4 x i32>* %BY
 store <4 x i32> %bz, <4 x i32>* %BZ
 store <4 x i32> %ca, <4 x i32>* %CA
 store <4 x i32> %cb, <4 x i32>* %CB
 store <4 x i32> %cc, <4 x i32>* %CC
 store <4 x i32> %cd, <4 x i32>* %CD
 store <4 x i32> %ce, <4 x i32>* %CE
 store <4 x i32> %cf, <4 x i32>* %CF
 store <4 x i32> %cg, <4 x i32>* %CG
 store <4 x i32> %ch, <4 x i32>* %CH
 store <4 x i32> %ci, <4 x i32>* %CI
 store <4 x i32> %cj, <4 x i32>* %CJ
 store <4 x i32> %ck, <4 x i32>* %CK
 store <4 x i32> %cl, <4 x i32>* %CL
 ret void
}
