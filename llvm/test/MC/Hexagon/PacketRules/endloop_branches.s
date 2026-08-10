# RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 | FileCheck %s

{ jump unknown
}:endloop0
# CHECK: 4:1: error: Branches cannot be in a packet with hardware loops

{ jump unknown
}:endloop1
# CHECK: 8:1: error: Branches cannot be in a packet with hardware loops

{ call unknown
}:endloop0
# CHECK: 12:1: error: Branches cannot be in a packet with hardware loops

{ dealloc_return
}:endloop0
# CHECK: 16:1: error: Branches cannot be in a packet with hardware loops
