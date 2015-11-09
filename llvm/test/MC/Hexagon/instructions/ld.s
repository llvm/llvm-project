# RUN: llvm-mc -triple hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.5 LD
# XFAIL: *

# Load doubleword
# CHECK: 90 ff d5 3a
r17:16 = memd(r21 + r31<<#3)
# CHECK: b0 c2 c0 49
r17:16 = memd(#168)
# CHECK: 02 40 00 00
# CHECK-NEXT: 10 c5 c0 49
r17:16 = memd(##168)
# CHECK: d0 c0 d5 91
r17:16 = memd(r21 + #48)
# CHECK: b0 e0 d5 99
r17:16 = memd(r21 ++ #40:circ(m1))
# CHECK: 10 e2 d5 99
r17:16 = memd(r21 ++ I:circ(m1))
# CHECK: 00 40 00 00
# CHECK-NEXT: 70 d7 d5 9b
r17:16 = memd(r21 = ##31)
# CHECK: b0 c0 d5 9b
r17:16 = memd(r21++#40)
# CHECK: 10 e0 d5 9d
r17:16 = memd(r21++m1)
# CHECK: 10 e0 d5 9f
r17:16 = memd(r21 ++ m1:brev)

# Load doubleword conditionally
# CHECK: f0 ff d5 30
if (p3) r17:16 = memd(r21+r31<<#3)
# CHECK: f0 ff d5 31
if (!p3) r17:16 = memd(r21+r31<<#3)
# CHECK: 03 40 45 85
# CHECK-NEXT: f0 ff d5 32
{ p3 = r5
  if (p3.new) r17:16 = memd(r21+r31<<#3) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f0 ff d5 33
{ p3 = r5
  if (!p3.new) r17:16 = memd(r21+r31<<#3) }
# CHECK: 70 d8 d5 41
if (p3) r17:16 = memd(r21 + #24)
# CHECK: 03 40 45 85
# CHECK-NEXT: 70 d8 d5 43
{ p3 = r5
  if (p3.new) r17:16 = memd(r21 + #24) }
# CHECK: 70 d8 d5 45
if (!p3) r17:16 = memd(r21 + #24)
# CHECK: 03 40 45 85
# CHECK-NEXT: 70 d8 d5 47
{ p3 = r5
  if (!p3.new) r17:16 = memd(r21 + #24) }
# CHECK: b0 e6 d5 9b
if (p3) r17:16 = memd(r21++#40)
# CHECK: b0 ee d5 9b
if (!p3) r17:16 = memd(r21++#40)
# CHECK: 03 40 45 85
# CHECK-NEXT: b0 f6 d5 9b
{ p3 = r5
  if (p3.new) r17:16 = memd(r21++#40) }
# CHECK: 03 40 45 85
# CHECK-NEXT: b0 fe d5 9b
{ p3 = r5
  if (!p3.new) r17:16 = memd(r21++#40) }

# Load byte
# CHECK: 91 ff 15 3a
r17 = memb(r21 + r31<<#3)
# CHECK: b1 c2 00 49
r17 = memb(#21)
# CHECK: 00 40 00 00
# CHECK-NEXT: b1 c2 00 49
r17 = memb(##21)
# CHECK: f1 c3 15 91
r17 = memb(r21 + #31)
# CHECK: b1 e0 15 99
r17 = memb(r21 ++ #5:circ(m1))
# CHECK: 11 e2 15 99
r17 = memb(r21 ++ I:circ(m1))
# CHECK: 00 40 00 00
# CHECK-NEXT: 71 d7 15 9b
r17 = memb(r21 = ##31)
# CHECK: b1 c0 15 9b
r17 = memb(r21++#5)
# CHECK: 11 e0 15 9d
r17 = memb(r21++m1)
# CHECK: 11 e0 15 9f
r17 = memb(r21 ++ m1:brev)

# Load byte conditionally
# CHECK: f1 ff 15 30
if (p3) r17 = memb(r21+r31<<#3)
# CHECK: f1 ff 15 31
if (!p3) r17 = memb(r21+r31<<#3)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 15 32
{ p3 = r5
  if (p3.new) r17 = memb(r21+r31<<#3) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 15 33
{ p3 = r5
  if (!p3.new) r17 = memb(r21+r31<<#3) }
# CHECK: 91 dd 15 41
if (p3) r17 = memb(r21 + #44)
# CHECK: 03 40 45 85
# CHECK-NEXT: 91 dd 15 43
{ p3 = r5
  if (p3.new) r17 = memb(r21 + #44) }
# CHECK: 91 dd 15 45
if (!p3) r17 = memb(r21 + #44)
# CHECK: 03 40 45 85
# CHECK-NEXT: 91 dd 15 47
{ p3 = r5
  if (!p3.new) r17 = memb(r21 + #44) }
# CHECK: b1 e6 15 9b
if (p3) r17 = memb(r21++#5)
# CHECK: b1 ee 15 9b
if (!p3) r17 = memb(r21++#5)
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 f6 15 9b
{ p3 = r5
  if (p3.new) r17 = memb(r21++#5) }
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 fe 15 9b
{ p3 = r5
  if (!p3.new) r17 = memb(r21++#5) }

# Load byte into shifted vector
# CHECK: f0 c3 95 90
r17:16 = memb_fifo(r21 + #31)
# CHECK: b0 e0 95 98
r17:16 = memb_fifo(r21 ++ #5:circ(m1))
# CHECK: 10 e2 95 98
r17:16 = memb_fifo(r21 ++ I:circ(m1))

# Load half into shifted vector
# CHECK: f0 c3 55 90
r17:16 = memh_fifo(r21 + #62)
# CHECK: b0 e0 55 98
r17:16 = memh_fifo(r21 ++ #10:circ(m1))
# CHECK: 10 e2 55 98
r17:16 = memh_fifo(r21 ++ I:circ(m1))

# Load halfword
# CHECK: 91 ff 55 3a
r17 = memh(r21 + r31<<#3)
# CHECK: b1 c2 40 49
r17 = memh(#42)
# CHECK: 00 40 00 00
# CHECK-NEXT: 51 c5 40 49
r17 = memh(##42)
# CHECK: f1 c3 55 91
r17 = memh(r21 + #62)
# CHECK: b1 e0 55 99
r17 = memh(r21 ++ #10:circ(m1))
# CHECK: 11 e2 55 99
r17 = memh(r21 ++ I:circ(m1))
# CHECK: 00 40 00 00
# CHECK-NEXT: 71 d7 55 9b
r17 = memh(r21 = ##31)
# CHECK: b1 c0 55 9b
r17 = memh(r21++#10)
# CHECK: 11 e0 55 9d
r17 = memh(r21++m1)
# CHECK: 11 e0 55 9f
r17 = memh(r21 ++ m1:brev)

# Load halfword conditionally
# CHECK: f1 ff 55 30
if (p3) r17 = memh(r21+r31<<#3)
# CHECK: f1 ff 55 31
if (!p3) r17 = memh(r21+r31<<#3)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 55 32
{ p3 = r5
  if (p3.new) r17 = memh(r21+r31<<#3) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 55 33
{ p3 = r5
  if (!p3.new) r17 = memh(r21+r31<<#3) }
# CHECK: b1 e6 55 9b
if (p3) r17 = memh(r21++#10)
# CHECK: b1 ee 55 9b
if (!p3) r17 = memh(r21++#10)
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 f6 55 9b
{ p3 = r5
  if (p3.new) r17 = memh(r21++#10) }
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 fe 55 9b
{ p3 = r5
  if (!p3.new) r17 = memh(r21++#10) }
# CHECK: f1 db 55 41
if (p3) r17 = memh(r21 + #62)
# CHECK: f1 db 55 45
if (!p3) r17 = memh(r21 + #62)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 db 55 43
{ p3 = r5
  if (p3.new) r17 = memh(r21 + #62) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 db 55 47
{ p3 = r5
  if (!p3.new) r17 = memh(r21 + #62) }

# Load unsigned byte
# CHECK: 91 ff 35 3a
r17 = memub(r21 + r31<<#3)
# CHECK: b1 c2 20 49
r17 = memub(#21)
# CHECK: 00 40 00 00
# CHECK-NEXT: b1 c2 20 49
r17 = memub(##21)
# CHECK: f1 c3 35 91
r17 = memub(r21 + #31)
# CHECK: b1 e0 35 99
r17 = memub(r21 ++ #5:circ(m1))
# CHECK: 11 e2 35 99
r17 = memub(r21 ++ I:circ(m1))
# CHECK: 00 40 00 00
# CHECK-NEXT: 71 d7 35 9b
r17 = memub(r21 = ##31)
# CHECK: b1 c0 35 9b
r17 = memub(r21++#5)
# CHECK: 11 e0 35 9d
r17 = memub(r21++m1)
# CHECK: 11 e0 35 9f
r17 = memub(r21 ++ m1:brev)

# Load unsigned byte conditionally
# CHECK: f1 ff 35 30
if (p3) r17 = memub(r21+r31<<#3)
# CHECK: f1 ff 35 31
if (!p3) r17 = memub(r21+r31<<#3)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 35 32
{ p3 = r5
  if (p3.new) r17 = memub(r21+r31<<#3) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 35 33
{ p3 = r5
  if (!p3.new) r17 = memub(r21+r31<<#3) }
# CHECK: f1 db 35 41
if (p3) r17 = memub(r21 + #31)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 db 35 43
{ p3 = r5
  if (p3.new) r17 = memub(r21 + #31) }
# CHECK: f1 db 35 45
if (!p3) r17 = memub(r21 + #31)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 db 35 47
{ p3 = r5
  if (!p3.new) r17 = memub(r21 + #31) }
# CHECK: b1 e6 35 9b
if (p3) r17 = memub(r21++#5)
# CHECK: b1 ee 35 9b
if (!p3) r17 = memub(r21++#5)
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 f6 35 9b
{ p3 = r5
  if (p3.new) r17 = memub(r21++#5) }
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 fe 35 9b
{ p3 = r5
  if (!p3.new) r17 = memub(r21++#5) }

# Load unsigned halfword
# CHECK: 91 ff 75 3a
r17 = memuh(r21 + r31<<#3)
# CHECK: b1 c2 60 49
r17 = memuh(#42)
# CHECK: 00 40 00 00
# CHECK-NEXT: 51 c5 60 49
r17 = memuh(##42)
# CHECK: b1 c2 75 91
r17 = memuh(r21 + #42)
# CHECK: b1 e0 75 99
r17 = memuh(r21 ++ #10:circ(m1))
# CHECK: 11 e2 75 99
r17 = memuh(r21 ++ I:circ(m1))
# CHECK: 00 40 00 00
# CHECK-NEXT: 71 d7 75 9b
r17 = memuh(r21 = ##31)
# CHECK: b1 c0 75 9b
r17 = memuh(r21++#10)
# CHECK: 11 e0 75 9d
r17 = memuh(r21++m1)
# CHECK: 11 e0 75 9f
r17 = memuh(r21 ++ m1:brev)

# Load unsigned halfword conditionally
# CHECK: f1 ff 75 30
if (p3) r17 = memuh(r21+r31<<#3)
# CHECK: f1 ff 75 31
if (!p3) r17 = memuh(r21+r31<<#3)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 75 32
{ p3 = r5
  if (p3.new) r17 = memuh(r21+r31<<#3) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 75 33
{ p3 = r5
  if (!p3.new) r17 = memuh(r21+r31<<#3) }
# CHECK: b1 da 75 41
if (p3) r17 = memuh(r21 + #42)
# CHECK: b1 da 75 45
if (!p3) r17 = memuh(r21 + #42)
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 da 75 43
{ p3 = r5
  if (p3.new) r17 = memuh(r21 + #42) }
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 da 75 47
{ p3 = r5
  if (!p3.new) r17 = memuh(r21 + #42) }
# CHECK: b1 e6 75 9b
if (p3) r17 = memuh(r21++#10)
# CHECK: b1 ee 75 9b
if (!p3) r17 = memuh(r21++#10)
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 f6 75 9b
{ p3 = r5
  if (p3.new) r17 = memuh(r21++#10) }
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 fe 75 9b
{ p3 = r5
  if (!p3.new) r17 = memuh(r21++#10) }

# Load word
# CHECK: 91 ff 95 3a
r17 = memw(r21 + r31<<#3)
# CHECK: b1 c2 80 49
r17 = memw(#84)
# CHECK: 01 40 00 00
# CHECK-NEXT: 91 c2 80 49
r17 = memw(##84)
# CHECK: b1 c2 95 91
r17 = memw(r21 + #84)
# CHECK: b1 e0 95 99
r17 = memw(r21 ++ #20:circ(m1))
# CHECK: 11 e2 95 99
r17 = memw(r21 ++ I:circ(m1))
# CHECK: 00 40 00 00
# CHECK-NEXT: 71 d7 95 9b
r17 = memw(r21 = ##31)
# CHECK: b1 c0 95 9b
r17 = memw(r21++#20)
# CHECK: 11 e0 95 9d
r17 = memw(r21++m1)
# CHECK: 11 e0 95 9f
r17 = memw(r21 ++ m1:brev)

# Load word conditionally
# CHECK: f1 ff 95 30
if (p3) r17 = memw(r21+r31<<#3)
# CHECK: f1 ff 95 31
if (!p3) r17 = memw(r21+r31<<#3)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 95 32
{ p3 = r5
  if (p3.new) r17 = memw(r21+r31<<#3) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 95 33
{ p3 = r5
  if (!p3.new) r17 = memw(r21+r31<<#3) }
# CHECK: b1 da 95 41
if (p3) r17 = memw(r21 + #84)
# CHECK: b1 da 95 45
if (!p3) r17 = memw(r21 + #84)
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 da 95 43
{ p3 = r5
  if (p3.new) r17 = memw(r21 + #84) }
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 da 95 47
{ p3 = r5
  if (!p3.new) r17 = memw(r21 + #84) }
# CHECK: b1 e6 95 9b
if (p3) r17 = memw(r21++#20)
# CHECK: b1 ee 95 9b
if (!p3) r17 = memw(r21++#20)
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 f6 95 9b
{ p3 = r5
  if (p3.new) r17 = memw(r21++#20) }
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 fe 95 9b
{ p3 = r5
  if (!p3.new) r17 = memw(r21++#20) }

# Deallocate stack frame
# CHECK: 1e c0 1e 90
deallocframe

# Deallocate stack frame and return
# CHECK: 1e c0 1e 96
dealloc_return
# CHECK: 03 40 45 85
# CHECK-NEXT: 1e cb 1e 96
{ p3 = r5
  if (p3.new) dealloc_return:nt }
# CHECK: 1e d3 1e 96
if (p3) dealloc_return
# CHECK: 03 40 45 85
# CHECK-NEXT: 1e db 1e 96
{ p3 = r5
  if (p3.new) dealloc_return:t }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1e eb 1e 96
{ p3 = r5
  if (!p3.new) dealloc_return:nt }
# CHECK: 1e f3 1e 96
if (!p3) dealloc_return
# CHECK: 03 40 45 85
# CHECK-NEXT: 1e fb 1e 96
{ p3 = r5
  if (!p3.new) dealloc_return:t }

# Load and unpack bytes to halfwords
# CHECK: f1 c3 35 90
r17 = membh(r21 + #62)
# CHECK: f1 c3 75 90
r17 = memubh(r21 + #62)
# CHECK: f0 c3 b5 90
r17:16 = memubh(r21 + #124)
# CHECK: f0 c3 f5 90
r17:16 = membh(r21 + #124)
# CHECK: b1 e0 35 98
r17 = membh(r21 ++ #10:circ(m1))
# CHECK: 11 e2 35 98
r17 = membh(r21 ++ I:circ(m1))
# CHECK: b1 e0 75 98
r17 = memubh(r21 ++ #10:circ(m1))
# CHECK: 11 e2 75 98
r17 = memubh(r21 ++ I:circ(m1))
# CHECK: b0 e0 f5 98
r17:16 = membh(r21 ++ #20:circ(m1))
# CHECK: 10 e2 f5 98
r17:16 = membh(r21 ++ I:circ(m1))
# CHECK: b0 e0 b5 98
r17:16 = memubh(r21 ++ #20:circ(m1))
# CHECK: 10 e2 b5 98
r17:16 = memubh(r21 ++ I:circ(m1))
# CHECK: 00 40 00 00
# CHECK-NEXT: 71 d7 35 9a
r17 = membh(r21 = ##31)
# CHECK: b1 c0 35 9a
r17 = membh(r21++#10)
# CHECK: 00 40 00 00
# CHECK-NEXT: 71 d7 75 9a
r17 = memubh(r21 = ##31)
# CHECK: b1 c0 75 9a
r17 = memubh(r21++#10)
# CHECK: 00 40 00 00
# CHECK-NEXT: 70 d7 b5 9a
r17:16 = memubh(r21 = ##31)
# CHECK: b0 c0 b5 9a
r17:16 = memubh(r21++#20)
# CHECK: 00 40 00 00
# CHECK-NEXT: 70 d7 f5 9a
r17:16 = membh(r21 = ##31)
# CHECK: b0 c0 f5 9a
r17:16 = membh(r21++#20)
# CHECK: 00 40 00 00
# CHECK-NEXT: f1 f7 35 9c
r17 = membh(r21<<#3 + ##31)
# CHECK: 11 e0 35 9c
r17 = membh(r21++m1)
# CHECK: 00 40 00 00
# CHECK-NEXT: f1 f7 75 9c
r17 = memubh(r21<<#3 + ##31)
# CHECK: 11 e0 75 9c
r17 = memubh(r21++m1)
# CHECK: 00 40 00 00
# CHECK-NEXT: f0 f7 f5 9c
r17:16 = membh(r21<<#3 + ##31)
# CHECK: 10 e0 f5 9c
r17:16 = membh(r21++m1)
# CHECK: 00 40 00 00
# CHECK-NEXT: f0 f7 b5 9c
r17:16 = memubh(r21<<#3 + ##31)
# CHECK: 11 e0 35 9c
r17 = membh(r21++m1)
# CHECK: 11 e0 75 9c
r17 = memubh(r21++m1)
# CHECK: 10 e0 f5 9c
r17:16 = membh(r21++m1)
# CHECK: 10 e0 b5 9c
r17:16 = memubh(r21++m1)
# CHECK: 11 e0 35 9e
r17 = membh(r21 ++ m1:brev)
# CHECK: 11 e0 75 9e
r17 = memubh(r21 ++ m1:brev)
# CHECK: 10 e0 b5 9e
r17:16 = memubh(r21 ++ m1:brev)
# CHECK: 10 e0 f5 9e
r17:16 = membh(r21 ++ m1:brev)
