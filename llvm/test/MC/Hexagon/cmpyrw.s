# RUN: llvm-mc -triple=hexagon -mv67t -filetype=obj %s | llvm-objdump --mcpu=hexagonv67t --mattr=+audio -d - | FileCheck %s
r23:22 = cmpyrw(r15:14,r21:20*)
# CHECK:   r23:22 = cmpyrw(r15:14,r21:20*)
