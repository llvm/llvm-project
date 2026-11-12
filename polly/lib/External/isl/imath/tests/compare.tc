# Test ordinary signed comparisons
cmp:0,0:$#0
cmp:0,1:$#-1
cmp:1,0:$#1
cmp:-1,1:$#-1
cmp:1,-1:$#1
cmp:123456789012345678901234567890,123456789012345678901234567891:$#-1

# Test unsigned comparisons
cmpu:0,0:$#0
cmpu:0,1:$#-1
cmpu:1,0:$#1
cmpu:-1,1:$#0
cmpu:1,-1:$#0
cmpu:-25,15:$#1
cmpu:#x-ffffffffffffffff,#xfffffffffffffffe:$#1

# Test zero comparisons
cmpz:0:$#0
cmpz:-25:$#-1
cmpz:105:$#1

# Test small-value comparisons
cmpv:0,0:$#0
cmpv:0,1:$#-1
cmpv:1,0:$#1
cmpv:-1,1:$#-1
cmpv:1,-1:$#1
cmpv:499,108:$#1
cmpv:499,499:$#0
cmpv:499,-1024:$#1
