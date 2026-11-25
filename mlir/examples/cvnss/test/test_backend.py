# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import subprocess
import sys
import itertools
opt = sys.argv[1]

def run(ir, *flags, ok=True):
    p = subprocess.run([opt, *flags], input=ir, text=True, capture_output=True)
    if (p.returncode == 0) != ok:
        raise AssertionError(p.stderr or p.stdout)
    return p.stdout if ok else p.stderr

for h, m, l in itertools.product(range(-1, 2), repeat=3):
    n = 13 + 9*h + 3*m + l
    ir = f'''module {{
      func.func @pack() -> i32 {{
        %r = "cvnss.pack27"() {{hi = {h} : i32, mid = {m} : i32, lo = {l} : i32}} : () -> i32
        return %r : i32
      }}
    }}'''
    out = run(ir, "--cvnss-lower-trit")
    assert f"arith.constant {n} : i32" in out and "cvnss.pack27" not in out, out
    out = run(ir, "--cvnss-lower-trit", "--convert-arith-to-llvm", "--convert-func-to-llvm", "--reconcile-unrealized-casts")
    assert "llvm.func" in out and "arith.constant" not in out and "cvnss.pack27" not in out, out
    ir = f'''module {{
      func.func @unpack() -> (i32, i32, i32) {{
        %a, %b, %c = "cvnss.unpack27"() {{id = {n} : i32}} : () -> (i32, i32, i32)
        return %a, %b, %c : i32, i32, i32
      }}
    }}'''
    out = run(ir, "--cvnss-lower-trit")
    import re
    assert [int(x) for x in re.findall(r"arith.constant (-?\d+) : i32", out)] == [h,m,l], out

bad = 'module { %0 = "cvnss.pack27"() {hi = 2 : i32, mid = 0 : i32, lo = 0 : i32} : () -> i32 }'
assert "trits must" in run(bad, ok=False)
bad = 'module { %a, %b, %c = "cvnss.unpack27"() {id = 27 : i32} : () -> (i32, i32, i32) }'
assert "ID must" in run(bad, ok=False)
record = 'module { "cvnss.mapping"() {source = "abc", target = "abc", mode = "KHD", mapping_version = "fixture-v1", loss_policy = "report"} : () -> () }'
assert "cvnss.mapping" in run(record, "--cvnss-lower-trit")
assert "mode must" in run(record.replace('"KHD"', '"bad"'), ok=False)
print("PASS: 27 pack/unpack cases, lowering and negative verification")
