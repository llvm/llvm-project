#!/usr/bin/env python3
"""Fetch GitHub Actions workflow runs and report timing bottlenecks.

Requires GITHUB_TOKEN or GH_TOKEN env var.

Usage:
    python ci-analyzer.py --summary
    python ci-analyzer.py --workflow premerge.yaml --runs 10
    python ci-analyzer.py --workflow premerge.yaml --runs 5 -v --json out.json
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime, timezone

GITHUB_API = "https://api.github.com"
COL_WIDTH = 45


def get_token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("Error: Set GITHUB_TOKEN or GH_TOKEN.", file=sys.stderr)
        sys.exit(1)
    return token


def api_get(endpoint, token):
    url = f"{GITHUB_API}{endpoint}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "llvm-ci-analyzer",
    })
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"API error {e.code}: {e.reason} — {url}", file=sys.stderr)
        return {}


def parse_duration(started, completed):
    if not started or not completed:
        return 0.0
    start = datetime.fromisoformat(started.replace("Z", "+00:00"))
    end = datetime.fromisoformat(completed.replace("Z", "+00:00"))
    return (end - start).total_seconds() / 60.0


def list_workflows(repo, token):
    data = api_get(f"/repos/{repo}/actions/workflows?per_page=100", token)
    return data.get("workflows", [])


def get_workflow_runs(repo, workflow_file, token, count=10):
    endpoint = (f"/repos/{repo}/actions/workflows/{workflow_file}"
                f"/runs?per_page={count}&status=completed")
    data = api_get(endpoint, token)
    return data.get("workflow_runs", [])


def get_run_jobs(repo, run_id, token):
    data = api_get(f"/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100", token)
    return data.get("jobs", [])


def analyze_run(repo, run, token):
    run_id = run["id"]
    duration = parse_duration(run.get("run_started_at", ""), run.get("updated_at", ""))
    jobs = get_run_jobs(repo, run_id, token)

    job_timings = []
    for job in jobs:
        job_dur = parse_duration(job.get("started_at", ""), job.get("completed_at", ""))
        steps = []
        for step in job.get("steps", []):
            step_dur = parse_duration(step.get("started_at", ""), step.get("completed_at", ""))
            steps.append({
                "name": step.get("name", "?"),
                "duration_min": round(step_dur, 2),
                "conclusion": step.get("conclusion", "?"),
            })
        job_timings.append({
            "name": job.get("name", "?"),
            "duration_min": round(job_dur, 2),
            "conclusion": job.get("conclusion", "?"),
            "steps": steps,
        })

    return {
        "run_id": run_id,
        "url": run.get("html_url", ""),
        "conclusion": run.get("conclusion", "?"),
        "total_duration_min": round(duration, 2),
        "trigger_event": run.get("event", "?"),
        "branch": run.get("head_branch", "?"),
        "created_at": run.get("created_at", "?"),
        "jobs": job_timings,
    }


def print_run_summary(analysis, verbose=False):
    a = analysis
    print(f"\n  Run #{a['run_id']}  |  {a['conclusion'].upper()}"
          f"  |  {a['total_duration_min']:.1f} min"
          f"  |  {a['trigger_event']}  |  {a['branch']}")
    print(f"  {a['url']}")
    print(f"  {'Job':<{COL_WIDTH}} {'Time':>8} {'Result':>10}")
    print(f"  {'-' * 65}")
    for job in a["jobs"]:
        print(f"  {job['name'][:COL_WIDTH-1]:<{COL_WIDTH}} "
              f"{job['duration_min']:>6.1f}m {job['conclusion']:>10}")
        if verbose:
            for step in job["steps"]:
                if step["duration_min"] > 0.5:
                    print(f"    {step['name'][:COL_WIDTH-3]:<{COL_WIDTH}} "
                          f"{step['duration_min']:>6.1f}m {step['conclusion']:>10}")


def print_aggregate(analyses):
    if not analyses:
        print("\nNo completed runs found.")
        return

    durations = [a["total_duration_min"] for a in analyses]
    conclusions = [a["conclusion"] for a in analyses]
    n = len(analyses)
    n_ok = conclusions.count("success")
    n_fail = conclusions.count("failure")
    avg = sum(durations) / n
    lo, hi = min(durations), max(durations)

    job_times = defaultdict(list)
    for a in analyses:
        for job in a["jobs"]:
            job_times[job["name"]].append(job["duration_min"])

    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE ({n} runs)")
    print(f"{'=' * 70}")
    print(f"  Success rate:  {n_ok}/{n} ({100*n_ok//n}%)")
    print(f"  Duration:      avg {avg:.1f}m  /  min {lo:.1f}m  /  max {hi:.1f}m")
    if n_fail:
        print(f"  Failures:      {n_fail}")

    print("\n  Slowest jobs (by avg):")
    print(f"  {'Job':<{COL_WIDTH}} {'Avg':>7} {'Max':>7}")
    print(f"  {'-' * 61}")
    ranked = sorted(job_times.items(), key=lambda kv: sum(kv[1])/len(kv[1]), reverse=True)
    for name, times in ranked[:10]:
        a_t = sum(times) / len(times)
        m_t = max(times)
        bar = "#" * min(int(a_t / 3), 30)
        print(f"  {name[:COL_WIDTH-1]:<{COL_WIDTH}} {a_t:>5.1f}m {m_t:>5.1f}m  {bar}")

    if n_fail:
        print("\n  Failed runs:")
        for a in analyses:
            if a["conclusion"] == "failure":
                bad = [j["name"] for j in a["jobs"] if j["conclusion"] == "failure"]
                print(f"    #{a['run_id']}: {', '.join(bad) or '(unknown job)'}")

    print("\n  Recommendations:")
    if avg > 60:
        print(f"    - Avg {avg:.0f}m > 60m target. Consider sccache + selective CI (P1-P2)")
    if avg > 120:
        print(f"    - Avg {avg:.0f}m > 120m. Self-hosted runners would help (P7)")
    if n_fail / max(n, 1) > 0.2:
        print(f"    - {100*n_fail//n}% failure rate. Investigate flaky tests (P4)")
    if avg <= 60 and n_fail / max(n, 1) <= 0.2:
        print("    - Looking good.")
    print()

    return avg


def export_json(analyses, path):
    with open(path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(analyses),
            "runs": analyses,
        }, f, indent=2)
    print(f"Exported {len(analyses)} runs to {path}")


def main():
    p = argparse.ArgumentParser(description="LLVM CI Bottleneck Analyzer")
    p.add_argument("--repo", default="llvm/llvm-project")
    p.add_argument("--workflow", help="Workflow filename (e.g. premerge.yaml)")
    p.add_argument("--runs", type=int, default=10, help="Number of recent runs to fetch")
    p.add_argument("--summary", action="store_true", help="List all workflows")
    p.add_argument("--json", help="Export results to JSON file")
    p.add_argument("-v", "--verbose", action="store_true", help="Show per-step timings")
    args = p.parse_args()

    token = get_token()
    print(f"LLVM CI Analyzer — {args.repo}")

    if args.summary or not args.workflow:
        workflows = list_workflows(args.repo, token)
        if not workflows:
            print("No workflows found.")
            return
        print(f"\n{'#':<4} {'Name':<{COL_WIDTH}} {'State':<10} File")
        print("-" * 90)
        for i, wf in enumerate(workflows, 1):
            name = wf.get("name", "?")[:COL_WIDTH-1]
            state = wf.get("state", "?")
            path = wf.get("path", "?").replace(".github/workflows/", "")
            print(f"{i:<4} {name:<{COL_WIDTH}} {state:<10} {path}")
        print(f"\n{len(workflows)} workflows total")
        if not args.workflow:
            print("Use --workflow <filename> to analyze a specific one.")
        return

    print(f"Fetching last {args.runs} completed runs of '{args.workflow}'...")
    runs = get_workflow_runs(args.repo, args.workflow, token, args.runs)
    if not runs:
        print(f"No completed runs found for '{args.workflow}'.")
        return

    print(f"Analyzing {len(runs)} runs...")
    analyses = []
    for run in runs:
        a = analyze_run(args.repo, run, token)
        analyses.append(a)
        print_run_summary(a, verbose=args.verbose)

    print_aggregate(analyses)

    if args.json:
        export_json(analyses, args.json)


if __name__ == "__main__":
    main()
