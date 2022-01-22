========================================
LLVM Security Group Transparency Reports
========================================

This page lists the yearly LLVM Security group transparency reports.

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

* https://bugs.chromium.org/p/llvm/issues/detail?id=5
* https://bugs.chromium.org/p/llvm/issues/detail?id=11

Both issues were addressed with source changes: #5 in clangd/vscode-clangd, and
#11 in llvm-project.  No dedicated LLVM release was made for either.

We believe that with the publishing of this first annual transparency report,
the security group now has implemented all necessary processes for the group to
operate as promised. The group's processes can be improved further, and we do
expect further improvements to get implemented in 2022. Many of the potential
improvements end up being discussed on the `monthly public call on LLVM's
security group <https://llvm.org/docs/GettingInvolved.html#online-sync-ups>`_.

