# `amdgpu-waves-per-eu` and the full-workgroup-residency ABI requirement

## Context

PR https://github.com/llvm/llvm-project/pull/138284 received a review comment
stating that a 1024-lane workgroup is 16 waves, and *all waves of a workgroup
must be resident on one CU at the same time* (barriers and LDS depend on
this), which on gfx942 (4 SIMDs/CU) requires 4 waves/SIMD and therefore caps
register usage at 128 VGPRs.

### Is "all waves resident simultaneously" actually true?

Yes. This is a hard hardware requirement, not just a scheduling convenience:

- **Barriers (`s_barrier`)**: a barrier only completes once every wave of the
  workgroup has reached it. If some waves weren't resident yet (merely queued
  waiting for a slot), the resident waves would block forever on waves that
  aren't even running — deadlock. Hardware guarantees this can't happen by
  never dispatching a workgroup until *all* its waves can be concurrently
  resident.
- **LDS**: LDS allocation for a workgroup is CU-local, shared by all its
  waves, and reserved for the workgroup's whole lifetime on that CU. There's
  no mechanism for a wave to join that allocation later.

So "on one CU" means "occupying an execution slot on one of the CU's SIMDs
concurrently," not "assigned/queued but possibly still waiting." A
workgroup's occupancy requirement is a real admission constraint enforced by
hardware dispatch, which is why register-allocation budgets must respect it.

## Does `AMDGPUSubtarget::getEffectiveWavesPerEU` honor this?

File: `llvm/lib/CodeGen/AMDGPU/AMDGPUSubtarget.cpp` (actual path:
`llvm/lib/Target/AMDGPU/AMDGPUSubtarget.cpp:186`).

**Yes, in principle.** `Default.first` (line 194) is computed as
`getWavesPerEUForWorkGroup(FlatWorkGroupSizes.second)`, which in
`AMDGPUBaseInfo.cpp:1186-1190` is `ceil(wavesPerWorkgroup(MaxWGSize) /
numSIMDs)` — exactly the minimum waves-per-EU occupancy needed so that every
wave of one *maximum-sized* workgroup can be resident on the CU's SIMDs at
once. This directly encodes the barrier/LDS invariant. The validity check at
lines 200-203 rejects any `RequestedWavesPerEU.first` below that floor, so a
user cannot request so few min-waves-per-EU (i.e. so many VGPRs/wave) that a
full-size workgroup couldn't fit concurrently.

**But the enforcement is too blunt — it's all-or-nothing.** If *any* of the
four validity conditions fails (min too low, min too high, min > max, max >
HW max), the function discards the *entire* requested pair and substitutes
the full `Default` — not just the offending bound. Example: `Default = {4,
6}`, user requests `{2, 5}`. The `min=2` is invalid, so the result becomes
`{4, 6}`, silently discarding the user's otherwise-valid `max=5` cap too, even
though 5 was legal and the user had no issue with the max. Control over an
explicitly-requested maximum is lost for a reason (invalid minimum) unrelated
to it. Only when the *whole* request is already valid does line 207 do a
targeted clamp (`second` down to `Default.second`).

There's also a quieter issue at line 196:
`Default.first = std::min(Default.first, Default.second)`. If LDS usage is
high enough that even the achievable max occupancy (`Default.second`) is
below the occupancy required to fit a full max-size workgroup, this silently
relaxes the "hard floor" down to whatever's achievable, instead of flagging
that this function's LDS/workgroup-size combination can never legally launch
a max-size workgroup. This masks a genuine conflict rather than diagnosing
it.

## Proposed improvements (not yet implemented)

1. **Clamp per-bound instead of per-request.** Replace the all-or-nothing
   fallback with:
   - `Requested.first = std::max(Requested.first, Default.first)`
   - `Requested.second = std::clamp(Requested.second, Requested.first, Default.second)`

   This preserves as much of the user's intent as is legal instead of
   discarding a valid `max` just because the `min` was invalid.

2. **Diagnose infeasible requests instead of silently substituting.** Emit a
   `DiagnosticInfoUnsupported`/optimization remark when:
   - The requested range is infeasible given the subtarget/workgroup size, or
   - `Default.first > Default.second` (the LDS-conflict case above) — i.e.
     the function's LDS usage and max flat workgroup size combination can
     never satisfy the full-workgroup-residency requirement at all.

   Currently a user whose attribute conflicts with the ABI floor just gets a
   silently different answer, with no indication their attribute was
   overridden.

3. **Make the `amdgpu-flat-work-group-size` / `amdgpu-waves-per-eu` coupling
   explicit.** The real lever for lowering the *required* minimum
   waves-per-EU is `amdgpu-flat-work-group-size` (smaller max workgroup size
   needs fewer concurrently-resident waves), not `amdgpu-waves-per-eu`
   itself. Diagnosing the conflict (per #2) when a user sets a
   `waves-per-eu` minimum that's incompatible with the current
   `flat-work-group-size` would make this relationship clear, and point the
   user at the attribute they actually need to adjust.

No code changes have been made yet; this file is a working note summarizing
the discussion so it can be picked up later.
