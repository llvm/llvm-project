# Thread Safety Analysis try-held stack: forward work plan

Context: Clang's Thread Safety Analysis
(`clang/lib/Analysis/ThreadSafety.cpp`, tests in
`clang/test/SemaCXX/warn-thread-safety-*.cpp` and
`clang/test/Sema/warn-thread-safety-analysis.c`) in this llvm-project
checkout, on the local `users/vtjnash/tsa-tryheld-*` branch stack — the
try-held (ternary) capability-state feature for
`try_acquire_capability`, destined for upstream as stacked PRs. Build
and test in `build-claude/` (not `build/`). All items come out of the
review cycle recorded in `tsa-review-known-findings.md` (seven reviews,
2026-08-22 to 2026-08-28), which remains the evidence record: every item
below cites its entry there for the verified repro and rationale. Do not
duplicate evidence here — extend the findings file, then update this
plan.

The original plan's Phases 0-5 (message repairs, join-surviving
negatives/S1/D6, loop/join precision D1/D2/D4/D5, D3 + duplicate-warning
polish, S2 value-precise resolution, scoped lockables) are all DONE
(2026-08-27), and so is candidate C1 (pre-elision guard factories —
branch `tsa-tryheld-guard-factories`, sixth-review residue 9 FIXED);
their completion records live in the findings file's FIXED entries, the
commit messages themselves, and the memory notes; this file keeps only
open work.

Stack shape this plan targets (unpushed):

    main
     └─ users/vtjnash/tsa-tryheld-nfc-prep
      └─ users/vtjnash/tsa-tryheld-state          (2 commits: ternary + mixed-success)
       └─ users/vtjnash/tsa-tryheld-never-checked
        └─ users/vtjnash/tsa-tryheld-edge-resolve
         └─ users/vtjnash/tsa-tryheld-stored-results
          └─ users/vtjnash/tsa-tryheld-value-precise
           └─ users/vtjnash/tsa-tryheld-scoped
            └─ users/vtjnash/tsa-tryheld-guard-factories
             └─ users/vtjnash/tsa-tryheld-check-before-call (tip)

## C2 — Order-independent try-acquire recording — IN PROGRESS

Implemented 2026-08-28 as the top-of-stack branch
`users/vtjnash/tsa-tryheld-check-before-call` (one commit
ea46ed354559, "Thread Safety Analysis: Record try-acquire capabilities
before the lockset walk"), unpushed like the rest of the stack; open
review findings below keep it in progress.

What shipped, in brief: recordTryAcquireCalls() populates
TryAcquireCapsMap for every try-acquire CallExpr before the lockset
walk (replaying the variable map's saved per-statement contexts;
handleCall() is find-only for CallExprs, with the opposite-polarity
reconciliation extracted to reconcileTryAcquireCaps() and the
unconditional capabilities stored in the map entry; constructors still
record in-walk; AnyTryLockFacts set up front), so getEdgeLockset()
resolves check-before-call edges from block one. Its companion join
rule in intersectAndWarn() demotes mixed same-origin facts silently to
try-held at LEK_LockedSomeLoopIterations joins only (SealedEntry
guards the back-edge comparison; the one-sided form accepts only the
call's own real un-spent failure-edge negative). The originally
sketched Part 1/Part 2 split was corrected during planning: one
semantic change, one commit; the first-iteration success edge needs no
dead-edge tracking (the loop-head phi's constant operand already makes
the constant edge Ambiguous); the branch-join diagnose-eagerly policy
was NOT reopened. tryheld_loop_join_not_a_leak's in-loop warning
flipped clean, tryheld_retry_with_continue stays clean, six new tests,
whole corpus otherwise unchanged. Full mechanism, the
null-RebranchTryLock spent-veto trap, and test list: findings file, D6
retirement note (plus the D2 update).

Remaining C2 work — seventh-review findings (2026-08-28, evidence and
dispositions in the findings file's Seventh review section), to fix by
amending the check-before-call commit:

- F7.3 dedup the try-acquire attribute decode: extract a shared helper
  used by recordTryAcquireCall() and handleCall()'s constructor path
  (~30 duplicated lines that can drift).
- F7.4 gate the pre-pass: skip the per-element context replay in
  functions with no try-acquire call (cheap attr-presence scan first,
  or detection piggybacked on traverseCFG()) — today every function
  under -Wthread-safety pays O(#CFG elements).
- F7.5 hoist SameOriginFailureNegative's result (currently evaluated
  twice per accepting path, each a linear findLock scan) and fold the
  twin DemoteToTryHeld blocks.
- F7.6 the loop-join gates mix EntryLEK and ExitLEK across the two
  intersection loops, correct only while every loop-LEK caller passes
  identical LEKs: normalize, or document the invariant at the
  intersectAndWarn() declaration.
- Not queued: F7.1 (pre-pass records calls in walk-unreachable blocks —
  KNOWN TRADE-OFF, plan risk 1; revisit only on real noise, the full
  gate cannot precede the walk) and F7.2 (different-origin
  spent-negative veto width — REJECTED, exactly mirrors the existing
  RebranchVetoedByNegative strength; widening would be a joint change
  to both exemptions).

## Other open items (see the findings file for evidence and decisions)

- Sixth-review test-gap residues 2-7 (beta gating, C-mode coverage,
  multi-code `==`, `!=` against a nonzero code, GNU case ranges,
  pthread-style falsy code): candidates for one test-only commit —
  either distributed into the stack commits whose behavior each test
  pins (C-mode/switch gaps -> edge-resolve, value-code gaps ->
  value-precise) at a restack, or one test-only commit on top.
- Parameter tracking precision note (escaped-param resolution): seed
  ParmVarDecls into the LocalVariableMap at function entry so
  `void f(bool ok) { ok = mu.TryLock(); if (ok) ... }` resolves.
- Join-drop beta warning: DEFERRED by choice; an all-paths gate is
  implementable (see the findings decisions block); revisit on beta
  trial noise.
- Diagnostics inside CoverageOnly blocks: KEPT as policy; revisit only
  on real-world noise.
