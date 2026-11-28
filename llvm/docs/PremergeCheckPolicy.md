# Premerge Check Policy

## Overview

Premerge checks run on pull requests before they are merged. They catch issues
on common platforms before they land, which increases the quality of `main` and
reduces post-commit triage. This policy sets the minimum requirements a
premerge check must meet to be enabled and to stay enabled.

It applies to all premerge checks: the main build-and-test pipelines (for
example Linux and Windows), and others such as the code formatting check. Some
requirements matter less for some checks (speed for the formatting check, for
example), but every check should be able to meet them.

New checks must meet these requirements before they are enabled. Existing
checks that stop meeting them may be made optional or disabled.

## Requirements

| Area | Metric | Target | Measured over |
|------|--------|--------|---------------|
| Speed | Runs completing in under 1 hour | More than 90% | One week |
| Speed | Runs completing in under 2 hours | 100% | One week |
| Reliability | False positive rate | Less than 1% | One month |

These are aspirational targets. As a community, we strive to work with
operators to meet these targets collaboratively, but if we aren't able to meet
these goals consistently, the [LLVM infra area team](ProjectGovernance.md#active-area-teams) may
decide to disable CI resources that it determines are not creating value for
the community.

### Speed

Operators must provide public monitoring of runtime data. A subproject may
accept slower checks if it has consensus that the check is worth the cost (for
example, the libc++ pipeline).

### Reliability

A false positive is a failure unrelated to the commit under test, such as an
infrastructure failure or an unrelated flaky test. Operators should look for
these proactively, but community reports are also acceptable.

Reliability also depends on the quality of the underlying LLVM tests. If the
tests themselves are flaky, brittle, or otherwise sensitive, we expect to fix
the tests rather than disabling our CI. However, we can only do this
effectively if our CI operators provide the community with publicly visible
metrics and monitoring, or we won't be able to prioritize deflaking the least
reliable tests.

### Configurability

Checks must be maintainable by the community:

- The build/test environment must have a declarative definition, ideally a
  Dockerfile built from an LLVM-owned repository. Other definitions are
  acceptable if the community can see and edit them.
- Infrastructure and environment definitions (for example, Terraform) must be
  public and in an LLVM-owned repository (such as llvm-zorg), as far as
  possible.
- Changes to these definitions need code review. Maintainers with the necessary
  permissions (for example, access to a cloud project) must review within one
  week of being pinged, barring major holidays.
