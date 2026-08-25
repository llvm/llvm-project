#!/usr/bin/env python3
"""Integration test for llvm-advisor HTTP endpoints added by the remarks work."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error


def main():
    if len(sys.argv) < 5:
        print("usage: test_http_endpoints.py <llvm-advisor> <opt.yaml> <source-root> <capability-dir>", file=sys.stderr)
        sys.exit(1)

    advisor = sys.argv[1]
    yaml_path = sys.argv[2]
    source_root = sys.argv[3]
    cap_dir = sys.argv[4]

    store = tempfile.mkdtemp(prefix="advisor-lit-")

    # Import the base remark file.
    subprocess.run(
        [advisor, "import", yaml_path, "--store", store, "--source-root", source_root, "--capability-dir", cap_dir],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Import a second (candidate) remark file so compare has differences.
    # Sleep briefly so the two imports get distinct snapshot IDs.
    cand_path = os.path.join(os.path.dirname(yaml_path), "demangle.opt.yaml")
    if os.path.exists(cand_path):
        time.sleep(1)
        subprocess.run(
            [advisor, "import", cand_path, "--store", store, "--source-root", source_root, "--capability-dir", cap_dir],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    # Start server on an ephemeral port.
    proc = subprocess.Popen(
        [advisor, "serve", "--store", store, "--port", "0", "--capability-dir", cap_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    port = None
    for _ in range(50):
        line = proc.stderr.readline()
        if line:
            m = re.search(r"listening on 127\.0\.0\.1:(\d+)", line)
            if m:
                port = int(m.group(1))
                break
        time.sleep(0.1)

    if port is None:
        proc.kill()
        proc.wait()
        print("server did not report port", file=sys.stderr)
        print("stderr:", proc.stderr.read(), file=sys.stderr)
        sys.exit(1)

    base = f"http://127.0.0.1:{port}/api/v1"
    errors = []

    def get(path, expect_status=200):
        url = base + path
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = resp.read().decode("utf-8")
                if resp.status != expect_status:
                    errors.append(f"{url}: expected {expect_status}, got {resp.status}")
                return data
        except urllib.error.HTTPError as e:
            if e.code != expect_status:
                errors.append(f"{url}: expected {expect_status}, got {e.code}")
            return e.read().decode("utf-8")

    # Health endpoint.
    health = json.loads(get("/health"))
    if not health.get("data", {}).get("ok"):
        errors.append("health endpoint returned ok=false")

    # Capabilities should include the remarks capabilities.
    caps = json.loads(get("/capabilities"))
    cap_ids = {c["id"] for c in caps.get("data", [])}
    expected_caps = {
        "llvm.remarks.summary",
        "llvm.remarks.detail",
        "llvm.remarks.relational",
        "llvm.remarks.hotspot",
    }
    if not expected_caps.issubset(cap_ids):
        errors.append(f"missing capabilities: {expected_caps - cap_ids}")

    # Identify base (example) and candidate (demangle) snapshots by source files.
    snaps = json.loads(get("/snapshots"))
    snap_list = snaps.get("data", [])
    if len(snap_list) < 2:
        errors.append(f"expected >=2 snapshots, got {len(snap_list)}")
        base_id = cand_id = None
    else:
        base_id = cand_id = None
        for snap in snap_list:
            sid = snap.get("id")
            files_data = json.loads(get(f"/snapshots/{sid}/files")).get("data", [])
            paths = {f.get("path") for f in files_data}
            if "src/example.c" in paths and base_id is None:
                base_id = sid
            if "src/demangle.cpp" in paths and cand_id is None:
                cand_id = sid
        if base_id is None:
            errors.append("could not find snapshot containing src/example.c")
        if cand_id is None:
            errors.append("could not find snapshot containing src/demangle.cpp")

    # Relational endpoint on the base snapshot.
    if base_id:
        rel = json.loads(get(f"/snapshots/{base_id}/remarks/relational?limit=50"))
        rel_data = rel.get("data", {})
        if rel_data.get("count", 0) != 41:
            errors.append(f"expected 41 relational rows, got {rel_data.get('count')}")
        rel_strings = rel_data.get("strings", {})
        if "normalize" not in rel_strings.get("function", []):
            errors.append("'normalize' not in relational function strings")
        if "sum_of_squares" not in rel_strings.get("function", []):
            errors.append("'sum_of_squares' not in relational function strings")

        # Relational filtering by pass.
        rel_pass = json.loads(get(f"/snapshots/{base_id}/remarks/relational?pass=inline&limit=50"))
        if rel_pass.get("data", {}).get("count", 0) != 2:
            errors.append("relational pass filter did not return 2 rows")
        if "Inlined" not in rel_pass.get("data", {}).get("strings", {}).get("name", []):
            errors.append("relational pass filter missing Inlined")

        # Relational filtering by function.
        from urllib.parse import quote
        rel_fn = json.loads(get(f"/snapshots/{base_id}/remarks/relational?function={quote('normalize')}&limit=50"))
        if rel_fn.get("data", {}).get("count", 0) == 0:
            errors.append("relational function filter returned 0 rows")

    # Compare aggregate endpoint.
    if base_id and cand_id:
        cmp_agg = json.loads(get(f"/compare/{base_id}/{cand_id}"))
        cmp_agg_data = cmp_agg.get("data", {})
        if "match_summary" not in cmp_agg_data:
            errors.append("compare aggregate missing match_summary")

    # Compare function detail endpoint for a function only in the candidate.
    if base_id and cand_id:
        from urllib.parse import quote
        cmp_fn = json.loads(get(f"/compare/{base_id}/{cand_id}/remarks/{quote('entry(int, int)')}"))
        cmp_fn_data = cmp_fn.get("data", {})
        if cmp_fn_data.get("function") != "entry(int, int)":
            errors.append("compare function detail returned wrong function")
        if cmp_fn_data.get("after_total", 0) == 0:
            errors.append("compare function detail candidate total is zero")
        if not cmp_fn_data.get("added"):
            errors.append("compare function detail missing added entries for candidate-only function")

    # Source files endpoint on the base snapshot.
    file_paths = []
    if base_id:
        files = json.loads(get(f"/snapshots/{base_id}/files"))
        file_paths = [f["path"] for f in files.get("data", [])]
        if "src/example.c" not in file_paths:
            errors.append("source files missing src/example.c")

    # Source content endpoint.
    if base_id and file_paths:
        from urllib.parse import quote
        src = json.loads(get(f"/source?path={quote('src/example.c')}&snapshot_id={base_id}"))
        if "content" not in src.get("data", {}):
            errors.append("source endpoint missing content")

        # Source remarks endpoint.
        src_remarks = json.loads(get(f"/source/remarks?path={quote('src/example.c')}&snapshot_id={base_id}"))
        if "remarks" not in src_remarks.get("data", {}):
            errors.append("source/remarks endpoint missing remarks")

        # Source remarks filtering by pass.
        src_remarks_pass = json.loads(get(f"/source/remarks?path={quote('src/example.c')}&snapshot_id={base_id}&pass=prologepilog"))
        sr_pass_data = src_remarks_pass.get("data", {})
        if "remarks" not in sr_pass_data:
            errors.append("source/remarks pass filter missing remarks")
        else:
            pass_names = {r.get("pass") for r in sr_pass_data["remarks"]}
            if pass_names != {"prologepilog"}:
                errors.append(f"source/remarks pass filter returned passes {pass_names}")

    # Compare remarks endpoint.
    if base_id and cand_id:
        cmp = json.loads(get(f"/compare/{base_id}/{cand_id}/remarks?offset=0&limit=10"))
        if "total" not in cmp.get("data", {}):
            errors.append("compare/remarks endpoint missing total")

    # Stop server.
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    shutil.rmtree(store, ignore_errors=True)

    if errors:
        print("FAIL:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
