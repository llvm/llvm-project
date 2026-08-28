===========================
LLVM GitHub Actions Runners
===========================

.. contents::
   :local:

Overview
========

LLVM's GitHub Actions workflows run on two kinds of runners:

- *GitHub-hosted runners*, which GitHub provisions on demand for each job.
- *Self-hosted runners*, which are machines provided to the LLVM project and
  registered with the repository. They are used where GitHub-hosted runners are
  not sufficient, for example to provide hardware or operating systems that
  GitHub does not offer, or to provide additional capacity.

Self-hosted runners are organized into sets, and a job selects a set through its
``runs-on`` labels. Since these machines are shared across the project and are
available only in limited numbers, workflows that target them should be written
to use them efficiently and to avoid consuming capacity unnecessarily.

The rest of this document describes the self-hosted runner sets and the
constraints to keep in mind when writing workflows that target them.

Self-Hosted Linux Runners
=========================

This section is a work in progress.

Self-Hosted Windows Runners
===========================

This section is a work in progress.

Self-Hosted macOS Runners
=========================

Self-hosted runners running macOS arm64 are provided by Apple. These runners can be targeted
with the following expression ``runs-on: ["self-hosted", "macOS", "apple-runners"]``. Since
these runners have a limited capacity, please contact the infrastructure team before adding
new jobs that target these runners.

System Version and Architecture
-------------------------------

All self-hosted macOS runners run the same version of macOS. However, that version
is determined by the image used on the runners, which is not controllable from
the workflow file. These runners will be kept at the latest released (non-beta)
version of macOS, however jobs running on that infrastructure should not make
assumptions about the macOS version and should strive to be robust to OS version
changes.

All the self-hosted macOS runners run on Apple Silicon, however the exact chip
version can differ from runner to runner. It is not currently possible to target
a specific chip version.

Minimize Short-Lived Jobs
-------------------------

The macOS runners are relatively expensive to bring up and tear down. Avoid scheduling
trivial or short-lived work on these runners. For example, do not spin up a macOS runner
just to perform a cheap check such as determining whether any relevant files have changed.
Prefer inexpensive runners instead and only then dispatch a macOS job if testing is actually
required.

Selecting the Xcode Version
---------------------------

The macOS runners come with several versions of Xcode installed: the two latest releases of
Xcode and the latest beta. You can select the version of Xcode by pointing ``DEVELOPER_DIR``
to it. The toolchain (``clang``, ``xcrun``, the SDKs, and so on) is then taken from that
Xcode. This can be done in an early step that writes the variable to ``$GITHUB_ENV`` so that
it applies to all subsequent steps:

.. code-block:: yaml

  - name: Select Xcode
    run: echo "DEVELOPER_DIR=/Applications/Xcode_26.5.app/Contents/Developer" >> $GITHUB_ENV

No Passwordless ``sudo``
------------------------

The user that runs jobs on the macOS runners cannot use ``sudo``: there is no passwordless
sudo, and jobs have no way to supply a password. Any step that requires root privileges will
therefore fail.

Installing Tools via Homebrew
-----------------------------

When a job needs a tool that is not already present on the runner, install it with Homebrew.
Homebrew installs into a prefix owned by the runner account, so it does not require ``sudo``,
and it provides self-contained tools. Also make sure you update Homebrew before installing.
For example:

.. code-block:: yaml

  - name: Install dependencies
    run: |
      brew update
      brew install ninja cmake

Version-specific formulae (for example ``python@3.12``) can be used when a job
needs a particular version of a tool.
