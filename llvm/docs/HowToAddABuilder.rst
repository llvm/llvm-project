===================================================================
How To Add Your Build Configuration To LLVM Buildbot Infrastructure
===================================================================

Introduction
============

This document contains information about adding a build configuration and
buildslave to private slave builder to LLVM Buildbot Infrastructure.

Buildmasters
============

There are two buildmasters running.

* The main buildmaster at `<http://lab.llvm.org:8011>`_. All builders attached
  to this machine will notify commit authors every time they break the build.
* The staging buildbot at `<http://lab.llvm.org:8014>`_. All builders attached
  to this machine will be completely silent by default when the build is broken.
  Builders for experimental backends should generally be attached to this
  buildmaster.

Steps To Add Builder To LLVM Buildbot
=====================================
Volunteers can provide their build machines to work as build slaves to
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

#. Install buildslave (currently we are using buildbot version 0.8.5).
   Depending on the platform, buildslave could be available to download and
   install with your package manager, or you can download it directly from
   `<http://trac.buildbot.net>`_ and install it manually.

#. Create a designated user account, your buildslave will be running under,
   and set appropriate permissions.

#. Choose the buildslave root directory (all builds will be placed under
   it), buildslave access name and password the build master will be using
   to authenticate your buildslave.

#. Create a buildslave in context of that buildslave account.  Point it to
   the **lab.llvm.org** port **9990** (see `Buildbot documentation,
   Creating a slave
   <http://docs.buildbot.net/current/tutorial/firstrun.html#creating-a-slave>`_
   for more details) by running the following command:

    .. code-block:: bash

       $ buildslave create-slave <buildslave-root-directory> \
                    lab.llvm.org:9990 \
                    <buildslave-access-name> <buildslave-access-password>

   To point a slave to silent master please use lab.llvm.org:9994 instead
   of lab.llvm.org:9990.

#. Fill the buildslave description and admin name/e-mail.  Here is an
   example of the buildslave description::

       Windows 7 x64
       Core i7 (2.66GHz), 16GB of RAM

       g++.exe (TDM-1 mingw32) 4.4.0
       GNU Binutils 2.19.1
       cmake version 2.8.4
       Microsoft(R) 32-bit C/C++ Optimizing Compiler Version 16.00.40219.01 for 80x86

#. Make sure you can actually start the buildslave successfully. Then set
   up your buildslave to start automatically at the start up time.  See the
   buildbot documentation for help.  You may want to restart your computer
   to see if it works.

#. Send a patch which adds your build slave and your builder to
   `zorg <https://github.com/llvm/llvm-zorg>`_. Use the typical LLVM 
   `workflow <https://llvm.org/docs/Contributing.html#how-to-submit-a-patch>`_.

   * slaves are added to ``buildbot/osuosl/master/config/slaves.py``
   * builders are added to ``buildbot/osuosl/master/config/builders.py``

   Please make sure your builder name and its builddir are unique through the
   file.

   It is possible to allow email addresses to unconditionally receive
   notifications on build failure; for this you'll need to add an
   ``InformativeMailNotifier`` to ``buildbot/osuosl/master/config/status.py``.
   This is particularly useful for the staging buildmaster which is silent
   otherwise.

#. Send the buildslave access name and the access password directly to
   `Galina Kistanova <mailto:gkistanova@gmail.com>`_, and wait till she
   will let you know that your changes are applied and buildmaster is
   reconfigured.

#. Check the status of your buildslave on the `Waterfall Display
   <http://lab.llvm.org:8011/waterfall>`_ to make sure it is connected, and
   ``http://lab.llvm.org:8011/buildslaves/<your-buildslave-name>`` to see
   if administrator contact and slave information are correct.

#. Wait for the first build to succeed and enjoy.
