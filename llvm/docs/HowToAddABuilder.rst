===================================================================
How To Add Your Build Configuration To LLVM Buildbot Infrastructure
===================================================================

Introduction
============

This document contains information about adding a build configuration and
buildbot-worker to private worker builder to LLVM Buildbot Infrastructure.

Buildmasters
============

There are two buildmasters running.

* The main buildmaster at `<https://lab.llvm.org/buildbot>`_. All builders
  attached to this machine will notify commit authors every time they break
  the build.
* The staging buildmaster at `<https://lab.llvm.org/staging>`_. All builders
  attached to this machine will be completely silent by default when the build
  is broken.

In order to remain connected to the main buildmaster (and thus notify
developers of failures), a builbot must:

* Be building a supported configuration.  Builders for experimental backends
  should generally be attached to staging buildmaster.
* Be able to keep up with new commits to the main branch, or at a minimum
  recover to tip of tree within a couple of days of falling behind.

Additionally, we encourage all bot owners to point their bots towards the
staging master during maintenance windows, instability troubleshooting, and
such.

Roles & Expectations
====================

Each buildbot has an owner who is the responsible party for addressing problems
which arise with said buildbot.  We generally expect the bot owner to be
reasonably responsive.

For some bots, the ownership responsibility is split between a "resource owner"
who provides the underlying machine resource, and a "configuration owner" who
maintains the build configuration.  Generally, operational responsibility lies
with the "config owner".  We do expect "resource owners" - who are generally
the contact listed in a workers attributes - to proxy requests to the relevant
"config owner" in a timely manner.

Most issues with a buildbot should be addressed directly with a bot owner
via email.  Please CC `Galina Kistanova <mailto:gkistanova@gmail.com>`_.

Steps To Add Builder To LLVM Buildbot
=====================================
Volunteers can provide their build machines to work as build workers to
public LLVM Buildbot.

Here are the steps you can follow to do so:

#. Check the existing build configurations to make sure the one you are
   interested in is not covered yet or gets built on your computer much
   faster than on the existing one. We prefer faster builds so developers
   will get feedback sooner after changes get committed.

#. The computer you will be registering with the LLVM buildbot
   infrastructure should have all dependencies installed and you can
   actually build your configuration successfully. Please check what degree
   of parallelism (-j param) would give the fastest build.  You can build
   multiple configurations on one computer.

#. Install buildbot-worker (currently we are using buildbot version 2.8.5).
   Depending on the platform, buildbot-worker could be available to download and
   install with your package manager, or you can download it directly from
   `<http://trac.buildbot.net>`_ and install it manually.

#. Create a designated user account, your buildbot-worker will be running under,
   and set appropriate permissions.

#. Choose the buildbot-worker root directory (all builds will be placed under
   it), buildbot-worker access name and password the build master will be using
   to authenticate your buildbot-worker.

#. Create a buildbot-worker in context of that buildbot-worker account. Point it
   to the **lab.llvm.org** port **9990** (see `Buildbot documentation,
   Creating a worker
   <http://docs.buildbot.net/current/tutorial/firstrun.html#creating-a-worker>`_
   for more details) by running the following command:

    .. code-block:: bash

       $ buildbot-worker create-worker <buildbot-worker-root-directory> \
                    lab.llvm.org:9990 \
                    <buildbot-worker-access-name> \
                    <buildbot-worker-access-password>

   To point a worker to silent master please use lab.llvm.org:9994 instead
   of lab.llvm.org:9990.

#. Fill the buildbot-worker description and admin name/e-mail.  Here is an
   example of the buildbot-worker description::

       Windows 7 x64
       Core i7 (2.66GHz), 16GB of RAM

       g++.exe (TDM-1 mingw32) 4.4.0
       GNU Binutils 2.19.1
       cmake version 2.8.4
       Microsoft(R) 32-bit C/C++ Optimizing Compiler Version 16.00.40219.01 for 80x86

#. Make sure you can actually start the buildbot-worker successfully. Then set
   up your buildbot-worker to start automatically at the start up time.  See the
   buildbot documentation for help.  You may want to restart your computer
   to see if it works.

#. Send a patch which adds your build worker and your builder to
   `zorg <https://github.com/llvm/llvm-zorg>`_. Use the typical LLVM 
   `workflow <https://llvm.org/docs/Contributing.html#how-to-submit-a-patch>`_.

   * workers are added to ``buildbot/osuosl/master/config/workers.py``
   * builders are added to ``buildbot/osuosl/master/config/builders.py``

   Please make sure your builder name and its builddir are unique through the
   file.

   It is possible to allow email addresses to unconditionally receive
   notifications on build failure; for this you'll need to add an
   ``InformativeMailNotifier`` to ``buildbot/osuosl/master/config/status.py``.
   This is particularly useful for the staging buildmaster which is silent
   otherwise.

#. Send the buildbot-worker access name and the access password directly to
   `Galina Kistanova <mailto:gkistanova@gmail.com>`_, and wait till she
   will let you know that your changes are applied and buildmaster is
   reconfigured.

#. Check the status of your buildbot-worker on the `Waterfall Display
   <http://lab.llvm.org/buildbot/#/waterfall>`_ to make sure it is connected,
   and the `Workers Display <http://lab.llvm.org/buildbot/#/workers>`_ to see if
   administrator contact and worker information are correct.

#. Wait for the first build to succeed and enjoy.
