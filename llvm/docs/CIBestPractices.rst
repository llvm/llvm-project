======================
LLVM CI Best Practices
======================

Overview
========

This document contains a list of guidelines and best practices to use when
working on LLVM's CI systems. These are intended to keep our actions reliable,
consistent, and secure.

Github Actions Best Practices
=============================

This section contains information on best practices/guidelines when working on
LLVM's github actions workflows.

Disabling Jobs In Forks
-----------------------

There are many LLVM forks that exist, and we currently default to preventing
actions from running outside of the LLVM organization to prevent them from
running in forks. We default to this as actions running in forks are usually
not desired and only run by accident. In addition, many of our workflows
assume that they are operating within the main LLVM repository and break
otherwise.

Adhering to this best practice looks like adding the following to each of the
jobs specified within a workflow:

.. code-block:: yaml

  jobs:
    <job name>:
      if: github.repository_owner == 'llvm'

We choose to use ``github.repository_owner`` rather than ``github.repository``
to enable these workflows to run in forks inside the LLVM organization such as
the ClangIR fork.

There are some exceptions to this rule where ``github.repository`` might be
used when it makes sense to limit a workflow to only running in the main
monorepo repository. These include things like the issue subscriber and
release tasks, which should not run anywhere else.

Hash Pinning Dependencies
-------------------------

Github Actions allows the use of actions from other repositories as steps in
jobs. We take advantage of various actions for a variety of different tasks,
but especially tasks like checking out the repository, and
downloading/uploading build caches. These actions are typically versioned with
just a release, which looks like the following:

.. code-block:: yaml

  steps:
    - name: Checkout LLVM
      uses: actions/checkout@v4

However, it is best practice to specify an exact commit SHA from which to pull
the action from, noting the version in a comment:

We plan on revisiting this recommendation once Github's immutable actions have
been rolled out as GA.

.. code-block:: yaml

  steps:
    - name: Checkout LLVM
      uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2

This is beneficial for two reasons: reliability and security. Specifying an
exact SHA rather than just a major version ensures we end up running the same
action originally specified when the workflow as authored and/or updated,
and that no breaking changes sneak in from new versions of a workflow being
released. However, this effect could also be achieved by specifying an exact
dot release. The biggest reason to prefer hash pinned dependencies is security.
Release assets on Github are mutable, allowing an attacker to change the code
within a specific version of an action after the fact, potentially stealing
sensitive tokens and credentials. Hash pinning the dependencies prevents this
as the hash would change with the code.

Using Versioned Runner Images
-----------------------------

Github actions allows the use of either specifically versioned runner images
(e.g., ``ubuntu-22.04``), or just the latest runner image
(e.g., ``ubuntu-latest``). It is best practice to use explicitly versioned
runner images. This prevents breakages when Github rolls the latest runner
image to a new version with potentially breaking changes, instead allowing us
to explicitly opt-in to using the new image when we have done sufficient
testing to ensure that our existing workflows work as expected in the new
environment.

Top Level Read Permissions
--------------------------

The top of every workflow should specify that the job only has read
permissions:

.. code-block:: yaml

  permissions:
    contents: read

If specific jobs within the workflow need additional permissions, those
permissions should be added within the specific job. This practice locks down
all permissions by default and only enables them when needed, better enforcing
the principle of least privilege.

Ensuring Workflows Run on the Correct Events
--------------------------------------------

Github allows workflows to run on a multitude of events and it is important to
configure a workflow such that it triggers on the correct events. There are
two main best practices around events that trigger workflows:

1. Workflows that are designed to run on pull requests should not be
restricted by target branch. Restricting the target branch unnecessarily
will prevent any stacked PRs from being tested. ``pull_request`` events should
not contain a branches key.

2. Workflows that are designed to also trigger on push events (e.g., for
testing on ``main`` or one of the release branches) need to be restricted by
branch. While pushes to a fork will not trigger a workflow run due to the
``push`` event if the workflow already has its jobs disabled in forks
(described above), stacked PRs will end up running jobs twice if the ``push``
event does not have any branch restrictions. ``push`` events should have
their branches restricted at the very least to ``main`` and the release
branches as follows:

.. code-block:: yaml

  push:
    branches:
      - main
      - releases/*
