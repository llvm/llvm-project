========================================
LLVM Security Group Transparency Reports
========================================

This page lists the yearly LLVM Security Response group transparency reports.

The LLVM Security Response group started out as the LLVM security group, previous
year's transparency reports keep the original name.

Initially the Chromium issue tracker was used to record issues. This
component has been archived and is read-only. A GitHub
llvm/llvm-project issue has been created for each issue in the
Chromium issue tracker. All of these issues contain an attached PDF
with the content of the Chromium issue, and have the SecurityArchive
label.

Each Chromium issue has 3 URLs, the first is the original URL recorded in
previous transparency reports. The second is the redirect URL to the archive.
The third is to the GitHub archive issue.

2021
----

The :doc:`LLVM security group <Security>` was established on the 10th of July
2020 by the act of the `initial
commit <https://github.com/llvm/llvm-project/commit/7bf73bcf6d93>`_ describing
the purpose of the group and the processes it follows.  Many of the group's
processes were still not well-defined enough for the group to operate well.
Over the course of 2021, the key processes were defined well enough to enable
the group to operate reasonably well:

* We defined details on how to report security issues, see `this commit on
  20th of May 2021 <https://github.com/llvm/llvm-project/commit/c9dbaa4c86d2>`_
* We refined the nomination process for new group members, see `this
  commit on 30th of July 2021 <https://github.com/llvm/llvm-project/commit/4c98e9455aad>`_
* We started writing an annual transparency report (you're reading the 2021
  report here).

Over the course of 2021, we had 2 people leave the LLVM Security group and 4
people join.

In 2021, the security group received 13 issue reports that were made publicly
visible before 31st of December 2021.  The security group judged 2 of these
reports to be security issues:

* original: https://bugs.chromium.org/p/llvm/issues/detail?id=5
  redirect: https://issuetracker.google.com/issues/42410043 archive:
  https://github.com/llvm/llvm-project/issues/125709

* original: https://bugs.chromium.org/p/llvm/issues/detail?id=11
  redirect: https://issuetracker.google.com/issues/42410002 archive:
  https://github.com/llvm/llvm-project/issues/127644

Both issues were addressed with source changes: #5 in clangd/vscode-clangd, and
#11 in llvm-project.  No dedicated LLVM release was made for either.

We believe that with the publishing of this first annual transparency report,
the security group now has implemented all necessary processes for the group to
operate as promised. The group's processes can be improved further, and we do
expect further improvements to get implemented in 2022. Many of the potential
improvements end up being discussed on the `monthly public call on LLVM's
security group <https://llvm.org/docs/GettingInvolved.html#online-sync-ups>`_.


2022
----

In this section we report on the issues the group received in 2022, or on issues
that were received earlier, but were disclosed in 2022.

In 2022, the llvm security group received 15 issues that have been disclosed at
the time of writing this transparency report.

5 of these were judged to be security issues:

* https://bugs.chromium.org/p/llvm/issues/detail?id=17 reports a miscompile in LLVM
  that can result in the frame pointer and return address being overwritten. This
  was fixed. Redirect: https://issuetracker.google.com/issues/42410008 archive:
  https://github.com/llvm/llvm-project/issues/127645

* https://bugs.chromium.org/p/llvm/issues/detail?id=19 reports a vulnerability in
  `std::filesystem::remove_all` in libc++. This was fixed.
  Redirect: https://issuetracker.google.com/issues/42410010 archive:
  https://github.com/llvm/llvm-project/issues/127647

* https://bugs.chromium.org/p/llvm/issues/detail?id=23 reports a new Spectre
  gadget variant that Speculative Load Hardening (SLH) does not mitigate. No
  extension to SLH was implemented to also mitigate against this variant.
  Redirect: https://issuetracker.google.com/issues/42410015 archive:
  https://github.com/llvm/llvm-project/issues/127648

* https://bugs.chromium.org/p/llvm/issues/detail?id=30 reports missing memory
  safety protection on the (C++) exception handling path. A number of fixes
  were implemented. Redirect: https://issuetracker.google.com/issues/42410023
  archive: https://github.com/llvm/llvm-project/issues/127649

* https://bugs.chromium.org/p/llvm/issues/detail?id=33 reports the RETBLEED
  vulnerability. The outcome was clang growing a new security hardening feature
  `-mfunction-return=thunk-extern`, see https://reviews.llvm.org/D129572.
  Redirect: https://issuetracker.google.com/issues/42410026 archive:
  https://github.com/llvm/llvm-project/issues/127650


No dedicated LLVM releases were made for any of the above issues.

2023
----

In this section we report on the issues the group received in 2023, or on issues
that were received earlier, but were disclosed in 2023.

9 of these were judged to be security issues:

 * https://bugs.chromium.org/p/llvm/issues/detail?id=36 reports the presence of
   .git folder in https://llvm.org/.git. Redirect:
   https://issuetracker.google.com/issues/42410029 archive:
   https://github.com/llvm/llvm-project/issues/131841

 * https://bugs.chromium.org/p/llvm/issues/detail?id=66 reports the presence of a
   GitHub Personal Access token in a DockerHub imaage. Redirect
   https://issuetracker.google.com/issues/42410060 archive:
   https://github.com/llvm/llvm-project/issues/131846

 * https://bugs.chromium.org/p/llvm/issues/detail?id=42 reports a potential gap
   in the Armv8.1-m BTI protection, involving a combination of large switch statements
   and __builtin_unreachable() in the default case. Redirect:
   https://issuetracker.google.com/issues/42410035 archive:
   https://github.com/llvm/llvm-project/issues/131848

 * https://bugs.chromium.org/p/llvm/issues/detail?id=43 reports a dependency
   on an old version of xml2js with a CVE filed against it. Redirect:
   https://issuetracker.google.com/issues/42410036 archive:
   https://github.com/llvm/llvm-project/issues/131849

 * https://bugs.chromium.org/p/llvm/issues/detail?id=45 reports a number of
   dependencies that have had vulnerabilities reported against them. Redirect:
   https://issuetracker.google.com/issues/42410038 archive:
   https://github.com/llvm/llvm-project/issues/131851

 * https://bugs.chromium.org/p/llvm/issues/detail?id=46 is related to
   issue 43. Redirect https://issuetracker.google.com/issues/42410039 archive:
   https://github.com/llvm/llvm-project/issues/131852

 * https://bugs.chromium.org/p/llvm/issues/detail?id=48 reports a buffer overflow in
   std::format from -fexperimental-library. Redirect:
   https://issuetracker.google.com/issues/42410041 archive:
   https://github.com/llvm/llvm-project/issues/131856

 * https://bugs.chromium.org/p/llvm/issues/detail?id=54 reports a memory leak in
   basic_string move assignment when built with libc++ versions <=6.0 and run against
   newer libc++ shared/dylibs. Redirect:
   https://issuetracker.google.com/issues/42410047 archive:
   https://github.com/llvm/llvm-project/issues/131857

 * https://bugs.chromium.org/p/llvm/issues/detail?id=56 reports an out
   of bounds buffer store introduced by LLVM backends, that regressed
   due to a procedural oversight. Redirect
   https://issuetracker.google.com/issues/42410049 archive:
   https://github.com/llvm/llvm-project/issues/131858

No dedicated LLVM releases were made for any of the above issues.

Over the course of 2023 we had one person join the LLVM Security Group.

2024
----

.. |br| raw:: html

  <br/>


Introduction
^^^^^^^^^^^^

In the first half of 2024, LLVM used the Chromium issue tracker to enable
reporting security issues responsibly. We switched over to using GitHub's
"privately reporting a security vulnerability" workflow in the middle of 2024.

In previous years, our transparency reports were shorter, since the full
discussion on a security ticket in the Chromium issue tracker is fully visible
once disclosed. This is not the case with issues using GitHub's security
advisory workflow, so instead we give a longer description in this transparency
report, to make the relevant information on the ticket publicly available.

This transparency report doesn't necessarily mention all issues that were deemed
duplicates of other issues, or tickets only created to test the bug tracking
system.

Security issues fixed under a coordinated disclosure process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section lists the reported issues where we ended up implementing fixes
under a coordinated disclosure process. While we were still using the Chromium
issue tracker, we did not write security advisories for such issues. Since we
started using the GitHub issues tracker for security issues, we're now
publishing security advisories for those issues at
https://github.com/llvm/llvm-security-repo/security/advisories/.

1. “Unexpected behavior when using LTO and branch-protection together” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=58 |br|
   redirect: https://issuetracker.google.com/issues/42410051 |br|
   archive: https://github.com/llvm/llvm-project/issues/132185
2. “Security weakness in PCS for CMSE”
   (`CVE-2024-0151 <https://nvd.nist.gov/vuln/detail/CVE-2024-0151>`_) |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=68 |br|
   redirect: https://issuetracker.google.com/issues/42410062 |br|
   archive: https://github.com/llvm/llvm-project/issues/132186
3. “CMSE secure state may leak from stack to floating-point registers”
   (`CVE-2024-7883 <https://www.cve.org/cverecord?id=CVE-2024-7883>`_) |br|
   Details are available at
   `GHSA-wh65-j229-6wfp <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-wh65-j229-6wfp>`_

Supply chain security related issues and project services-related issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. “GitHub User Involved in xz backdoor may have attempted to change to clang in order to help hide the exploit” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=71 |br|
   redirect: https://issuetracker.google.com/issues/42410066 |br|
   archive: https://github.com/llvm/llvm-project/issues/132187
2. “llvmbot account suspended due to suspicious login” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=72 |br|
   redirect: https://issuetracker.google.com/issues/42410067 |br|
   archive: https://github.com/llvm/llvm-project/issues/132243
3. “.git Exposure” |br|
   GHSA-mr8r-vvrc-w6rq |br|
   The .git directory was accessible via web browsers under apt.llvm.org, a site
   used to serve Debian/Ubuntu nightly packages. This issue has been addressed
   by removing the directory, and is not considered a security issue for the
   compiler. The .git directory was an artifact of the CI job that maintained
   the apt website, and was mirroring an open-source project maintained on
   github (under opencollab/llvm-jenkins.debian.net). The issue is not believed
   to have leaked any non-public information.
4. “llvm/llvm-project repo potentially vulnerable to GITHUB\_TOKEN leaks” |br|
   GHSA-f5xj-84f9-mrw6 |br|
   GitHub access tokens were being leaked in artifacts generated by GitHub
   Actions workflows. The vulnerability was first reported publicly as
   ArtiPACKED, generally applicable to GitHub projects, leading to an audit of
   LLVM projects and the reporting of this security issue. LLVM contributors
   audited the workflows, found that the “release-binaries” workflow was
   affected, but only exposed tokens that were ephemeral and read-only, so was
   not deemed a privilege escalation concern. The workflow was fixed in a
   configuration change as PR
   `106310 <https://github.com/llvm/llvm-project/pull/106310>`_. Older exposed
   tokens all expired, and the issue is closed as resolved.
5. “RCE in Buildkite Pipeline” |br|
   GHSA-2j6q-qcfm-3wcx |br|
   A Buildkite CI pipeline (llvm-project/rust-llvm-integrate-prototype) allowed
   Remote Code Execution on the CI runner. The pipeline automatically runs a
   test job when PRs are filed on the rust-lang/rust repo, but those PRs point
   to user-controlled branches that could be maliciously modified. A security
   researcher reported the issue, and demonstrated it by modifying build scripts
   to expose the CI runner's internal cloud service access tokens. The issue has
   been addressed with internal configuration changes by owners of the Buildkite
   pipeline.

Issues deemed to not require coordinated action before disclosing publicly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. “Clang Address Sanitizer gives False Negative for Array Out of Bounds Compiled with Optimization” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=57 |br|
   redirect: https://issuetracker.google.com/issues/42410050 |br|
   archive: https://github.com/llvm/llvm-project/issues/132191
2. “Found exposed .svn folder” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=59 |br|
   redirect: https://issuetracker.google.com/issues/42410052
   archive: https://github.com/llvm/llvm-project/issues/132192
3. “Arbitrary code execution when combining SafeStack \+ dynamic stack allocations \+ \_\_builtin\_setjmp/longjmp” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=60 |br|
   redirect: https://issuetracker.google.com/issues/42410054
   archive: https://github.com/llvm/llvm-project/issues/132220
4. “RISC-V: Constants are allocated in writeable .sdata section” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=61 |br|
   redirect: https://issuetracker.google.com/issues/42410055 |br|
   archive: https://github.com/llvm/llvm-project/issues/132223
5. “Manifest File with Out-of-Date Dependencies with CVEs” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=62 |br|
   redirect: https://issuetracker.google.com/issues/42410056 |br|
   archive: https://github.com/llvm/llvm-project/issues/132225
6. “Non-const derived ctor should fail compilation when having a consteval base ctor” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=67 |br|
   redirect: https://issuetracker.google.com/issues/42410061 |br|
   archive: https://github.com/llvm/llvm-project/issues/132226
7. “Wrong assembly code generation. Branching to the corrupted "LR".” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=69 |br|
   redirect: https://issuetracker.google.com/issues/42410063 |br|
   archive: https://github.com/llvm/llvm-project/issues/132229
8. “Security bug report” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=70 |br|
   redirect: https://issuetracker.google.com/issues/42410065 |br|
   archive: https://github.com/llvm/llvm-project/issues/132233
9. “Using ASan with setuid binaries can lead to arbitrary file write and elevation of privileges” |br|
   Details are available at https://bugs.chromium.org/p/llvm/issues/detail?id=73 |br|
   redirect: https://issuetracker.google.com/issues/42410068 |br|
   archive: https://github.com/llvm/llvm-project/issues/132235
10. “Interesting bugs for bool variable in clang projects and aarch64 modes outputting inaccurate results.” |br|
    GHSA-w7qc-292v-5xh6 |br|
    The issue reported is on a source code example having undefined behaviour
    (UB), somewhat similar to this: https://godbolt.org/z/vo4P7bPYr.
    Therefore, this issue was closed as not a security issue in the compiler. |br|
    As part of the analysis on this issue, it was deemed useful to document this
    example of UB and similar cases to help users of compilers understand how UB
    in source code can lead to security issues. |br|
    We concluded that probably the best option to do so is to create a regular
    public issue at https://github.com/llvm/llvm-project/issues, with the same
    title as the security issue, and to attach a PDF (which should easily be
    created using a “print-to-pdf” method in the browser) containing all
    comments. Such public tickets probably need some consistent way to indicate
    they come from security issues that after analysis were deemed to be outside
    the LLVM threat model or weren't accepted as a
    needs-resolution-work-in-private security issue for other reasons. The LLVM
    Security Response group has so far not taken action to progress this idea. |br|
    There was also a suggestion of potentially adding a short section in
    https://llsoftsec.github.io/llsoftsecbook/#compiler-introduced-security-vulnerabilities
    that summarizes a short example showing that type aliasing UB can and is
    causing security vulnerabilities.
11. “llvm-libc qsort can use very large amounts of stack if an attacker can control its input list” |br|
    GHSA-gw5j-473x-p29m |br|
    If the llvm-libc `qsort` function is used in a context where its input list
    comes from an attacker, then the attacker can craft a list that causes
    `qsort`'s stack usage to be linear in the size of the input array,
    potentially overflowing the available memory region for the stack. |br|
    After discussion with stakeholders, including maintainers for llvm-libc, the
    conclusion was that this doesn't have to be processed as a security issue
    needing coordinated disclosure. An improvement to `qsort`'s implementation
    was implemented through pull request
    https://github.com/llvm/llvm-project/pull/110849.
12. “VersionFromVCS.cmake may leak secrets in released builds” |br|
    GHSA-rcw6-jqvr-fcrx |br|
    The LLVM build system may leak secrets of VCS configuration into release
    builds if the user clones the repo with an https link that contains their
    username and/or password. |br|
    Mitigations were implemented in
    https://github.com/llvm/llvm-project/pull/105220,
    https://github.com/llvm/llvm-project/commit/57dc09341e5eef758b1abce78822c51069157869.
    An issue was raised to suggest one more mitigation to be implemented at
    https://github.com/llvm/llvm-project/issues/109030.

Invalid issues
^^^^^^^^^^^^^^

The LLVM security group received 5 issues which were created accidentally or
were not related to the LLVM project. The subject lines for these were:

* “Found this in my android”
* “\[Not a new security issue\] Continued discussion for GHSA-w7qc-292v-5xh6”
* “please delete it.”
* “Please help me to delete it.”
* “llvm code being used in malicious hacking of network and children's devices”

Furthermore, we had 2 tickets that were created to test the setup and workflow
as part of migrating to GitHub's “security advisory”-based reporting:

1. “Test if new draft security advisory gets emailed to LLVM security group” |br|
   GHSA-82m9-xvw3-rvpv
2. “Test that a non-admin can create an advisory (no vulnerability).” |br|
   GHSA-34gr-6c7h-cc93

2025
----

Introduction
^^^^^^^^^^^^

2025 was the first year all reports were submitted using Github. We report on
the issues the group received in 2025, or on issues that were received
earlier, but were disclosed in 2025.

We group the issues into the following categories:

1. Security issues fixed under a coordinated disclosure process (2 issues)
2. Supply chain security related issues and project services-related issues
   (2 issues)
3. Issues deemed to not require coordinated action before disclosing publicly
   (11 issues)
4. Invalid issues (5 issues)

In 2025, we received 2 invalid issues that we believe that have been created
automatically and 1 issue appeared to be created using generative AI. That
issue was considered to be invalid.

Security issues fixed under a coordinated disclosure process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section lists the reported issues where we ended up implementing fixes
under a coordinated disclosure process. The security advisories for those
issues can be found at
https://github.com/llvm/llvm-security-repo/security/advisories/.

1. “CMSE secure state may leak from stack to floating-point registers” |br|
   Details are available at
   `GHSA-wh65-j229-6wfp <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-wh65-j229-6wfp>`_
2. “Binary executable injection vulnerability in clang-linker-wrapper.exe” |br|
   Details are available at
   `GHSA-hrx2-grgx-9vhg <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-hrx2-grgx-9vhg>`_

Supply chain security related issues and project services-related issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. “Critical Supply Chain Vulnerability in RockstarGames/llvm-project
   (CVE-2025-30066)” |br|
   Details are available at
   `GHSA-3fq9-qcq4-8jjr <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-3fq9-qcq4-8jjr>`_ |br|
   The issue had already been fixed with commit
   `6616acd80cd91 <https://github.com/llvm/llvm-project/commit/6616acd80cd91a0075e3cd481bb9a6d82fd4ea9e>`_.
2. “CVE-2022-25883 and CVE-2022-3517 with respect to
   llvm-project/mlir/utils/vscode/package-lock.json” |br|
   Details are available at
   `GHSA-g72r-487m-m6hh <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-g72r-487m-m6hh>`_ |br|
   Packages have been updated with
   `PR 144479 <https://github.com/llvm/llvm-project/pull/144479>`_.

Issues deemed to not require coordinated action before disclosing publicly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. “Coroutine Frame-Oriented Programming: A new exploitation method using C++ coroutines” |br|
   Details are available at
   `GHSA-v8pv-j8f5-qqcg <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-v8pv-j8f5-qqcg>`_ |br|
   The researchers shared a new exploitation method that leverages the
   implementation of C++ routines. Their
   `paper <https://www.usenix.org/conference/usenixsecurity25/presentation/bajo>`_
   describing the technique has been published and is available publicly.
2. “Security Bug in String Assign Function (libc++)” |br|
   Details are available at
   `GHSA-m967-6j3p-jrwc <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-m967-6j3p-jrwc>`_ |br|
   There has been an agreement that the proof of concept had undefined
   behavior which makes it out of scope according to
   `the LLVM threat model <https://llvm.org/docs/Security.html#what-is-considered-a-security-issue>`_.
3. “\[clangd\] heap-use-after-free in clangd when generating diagnostics” |br|
   Details are available at
   `GHSA-5426-9r4h-7whf <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-5426-9r4h-7whf>`_ |br|
   It has been agreed this report fell out of scope because it was caused by
   untrusted inputs, as described in
   `the LLVM threat model <https://llvm.org/docs/Security.html#what-is-considered-a-security-issue>`_.
4. “A compiler optimization bug may cause signed integer overflow detection
   be bypassed” |br|
   Details are available at
   `GHSA-w6jm-h8j9-q33r <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-w6jm-h8j9-q33r>`_ |br|
   There has been an agreement that the PoC had undefined behavior which makes
   it out of scope according to
   `the LLVM threat model <https://llvm.org/docs/Security.html#what-is-considered-a-security-issue>`_.
5. “libomp: Crash (OOB Write / ASan BUS Error) involving omp\_init\_lock under
   high concurrency” |br|
   Details are available at
   `GHSA-cfhc-jxq2-97mf <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-cfhc-jxq2-97mf>`_ |br|
   The group agreed to close this as not a security issue because the code
   was written without taking into consideration the expectations from the
   OpenMP specification.
6. “\[MLIR\] head-use-after-free in mlir-lsp-server on completion request” |br|
   Details are available at
   `GHSA-8j9r-qc4r-q9fh <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-8j9r-qc4r-q9fh>`_ |br|
   This report fell out of scope because it was caused by untrusted inputs,
   as described in
   `the LLVM threat model <https://llvm.org/docs/Security.html#what-is-considered-a-security-issue>`_.
7. “\[clangd/clang\] heap-buffer-overflow in clang/lib/Sema/SemaExprCXX.cpp:9144” |br|
   Details are available at
   `GHSA-qq8q-r524-8vw9 <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-qq8q-r524-8vw9>`_ |br|
   This issue and the following 3 were concluded to be outside of the
   `LLVM threat model <https://llvm.org/docs/Security.html#what-is-considered-a-security-issue>`_.
8. “\[clangd\] heap-buffer-overflow in clang/lib/Sema/SemaExprCXX.cpp:8876” |br|
   Details are available at
   `GHSA-3xm9-vccr-fxx5 <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-3xm9-vccr-fxx5>`_
9. “\[clangd/clang\] heap-use-after-free at clang/Sema/Ownership.h:81” |br|
   Details are available at
   `GHSA-qj36-2p7g-83gv <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-qj36-2p7g-83gv>`_
10. “Clang 20.1.0 Compiler Internal Error (Crash) during AST Parsing of C++23” |br|
    Details are available at
    `GHSA-p2g2-89wf-7gcm <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-p2g2-89wf-7gcm>`_
11. “Compiler-induced non-constant-time code” |br|
    Details are available at
    `GHSA-627p-g235-23pm <https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-627p-g235-23pm>`_ |br|
    The reporters shared a pre-print article evaluating non-constant-time
    code generated by Clang. We all agreed this did not need coordinated
    disclosure because Clang offers no guarantees of constant-time code.

Invalid issues
^^^^^^^^^^^^^^

The LLVM security group received 5 issues which were created accidentally or
were not related to the LLVM project. The subject lines for these were:

1. “llvm-Bug”
2. “Potential Negative Number Used as Index”
3. “I was recently hacked... maybe you folks might know the dev?”
4. “ASP.NETconfiguration: Creating Debug Binary in ``[![Labelling new pull requests](https://github.com/llvm/llvm-project/actions/workflows/new-prs.yml/badge.svg?event=create)](https://github.com/llvm/llvm-project/actions/workflows``”
5. “ASP.NETconfiguration: Creating Debug Binary in ``[![Labelling new pull requests](https://github.com/llvm/llvm-project/actions/workflows/new-prs.yml/badge.svg?event=create)](https://github.com/llvm/llvm-project/actions/workflows``”
