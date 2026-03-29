# Mask operations

vsetvli x28, x0, e8, mf2, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmand.mm v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmand.mm v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmnand.mm v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmnand.mm v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmandn.mm v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmandn.mm v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmxor.mm v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmxor.mm v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmor.mm v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmor.mm v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmnor.mm v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmnor.mm v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmorn.mm v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmorn.mm v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e8, m1, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e8, m2, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e8, m4, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e8, m8, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e16, m1, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e16, m2, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e16, m4, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e16, m8, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e32, m1, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e32, m2, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e32, m4, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e32, m8, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e64, m1, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e64, m2, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e64, m4, tu, mu
vmxnor.mm v8, v8, v8
vsetvli x28, x0, e64, m8, tu, mu
vmxnor.mm v8, v8, v8

vsetvli x28, x0, e8, mf2, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e8, mf4, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e8, mf8, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e8, m1, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e8, m2, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e8, m4, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e8, m8, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e16, mf2, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e16, m8, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e32, m8, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e64, m1, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e64, m2, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e64, m4, tu, mu
vmsbf.m v8, v16
vsetvli x28, x0, e64, m8, tu, mu
vmsbf.m v8, v16

vsetvli x28, x0, e8, mf2, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e8, mf4, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e8, mf8, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e8, m1, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e8, m2, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e8, m4, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e8, m8, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e16, mf2, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e16, m8, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e32, m8, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e64, m1, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e64, m2, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e64, m4, tu, mu
vmsif.m v8, v16
vsetvli x28, x0, e64, m8, tu, mu
vmsif.m v8, v16

vsetvli x28, x0, e8, mf2, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e8, mf4, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e8, mf8, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e8, m1, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e8, m2, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e8, m4, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e8, m8, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e16, mf2, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e16, mf4, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e16, m1, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e16, m2, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e16, m4, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e16, m8, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e32, mf2, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e32, m1, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e32, m2, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e32, m4, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e32, m8, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e64, m1, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e64, m2, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e64, m4, tu, mu
vmsof.m v8, v16
vsetvli x28, x0, e64, m8, tu, mu
vmsof.m v8, v16

vsetvli x28, x0, e8, mf2, tu, mu
vid.v v8
vsetvli x28, x0, e8, mf4, tu, mu
vid.v v8
vsetvli x28, x0, e8, mf8, tu, mu
vid.v v8
vsetvli x28, x0, e8, m1, tu, mu
vid.v v8
vsetvli x28, x0, e8, m2, tu, mu
vid.v v8
vsetvli x28, x0, e8, m4, tu, mu
vid.v v8
vsetvli x28, x0, e8, m8, tu, mu
vid.v v8
vsetvli x28, x0, e16, mf2, tu, mu
vid.v v8
vsetvli x28, x0, e16, mf4, tu, mu
vid.v v8
vsetvli x28, x0, e16, m1, tu, mu
vid.v v8
vsetvli x28, x0, e16, m2, tu, mu
vid.v v8
vsetvli x28, x0, e16, m4, tu, mu
vid.v v8
vsetvli x28, x0, e16, m8, tu, mu
vid.v v8
vsetvli x28, x0, e32, mf2, tu, mu
vid.v v8
vsetvli x28, x0, e32, m1, tu, mu
vid.v v8
vsetvli x28, x0, e32, m2, tu, mu
vid.v v8
vsetvli x28, x0, e32, m4, tu, mu
vid.v v8
vsetvli x28, x0, e32, m8, tu, mu
vid.v v8
vsetvli x28, x0, e64, m1, tu, mu
vid.v v8
vsetvli x28, x0, e64, m2, tu, mu
vid.v v8
vsetvli x28, x0, e64, m4, tu, mu
vid.v v8
vsetvli x28, x0, e64, m8, tu, mu
vid.v v8

vsetvli x28, x0, e8, mf2, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e8, mf4, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e8, mf8, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e8, m1, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e8, m2, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e8, m4, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e8, m8, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e16, mf2, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e16, m1, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e16, m2, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e16, m4, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e16, m8, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e32, m1, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e32, m2, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e32, m4, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e32, m8, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e64, m1, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e64, m2, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e64, m4, tu, mu
vcpop.m x8, v8
vsetvli x28, x0, e64, m8, tu, mu
vcpop.m x8, v8

vsetvli x28, x0, e16, mf2, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e16, mf4, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e16, m1, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e16, m2, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e16, m4, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e16, m8, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e32, mf2, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e32, m1, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e32, m2, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e32, m4, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e32, m8, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e64, m1, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e64, m2, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e64, m4, tu, mu
vfirst.m x8, v8
vsetvli x28, x0, e64, m8, tu, mu
vfirst.m x8, v8
