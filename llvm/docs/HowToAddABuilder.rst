===================================================================
How To Add Your Build Configuration To LLVM Buildbot Infrastructure
===================================================================

Introduction
============

This document contains information about adding a build configuration and
buildbot worker to the LLVM Buildbot Infrastructure.

.. note:: The term "buildmaster" is used in this document to refer to the
  server that manages which builds are run and where. Though we would not
  normally choose to use "master" terminology, it is used in this document
  because it is the term that the Buildbot package currently
  `uses <https://github.com/buildbot/buildbot/issues/5382>`_.

Buildmasters
============

There are two buildmasters running.

* The main buildmaster at `<https://lab.llvm.org/buildbot>`_. All builders
  attached to this machine will notify commit authors every time they break
  the build.
* The staging buildmaster at `<https://lab.llvm.org/staging>`_. All builders
  attached to this machine will be completely silent by default when the build
  is broken. This buildmaster is reconfigured every two hours with any new
  commits from the llvm-zorg repository.

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
   infrastructure should have all dependencies installed and be able to
   build your configuration successfully. Please check what degree
   of parallelism (-j param) would give the fastest build.  You can build
   multiple configurations on one computer.

#. Install buildbot-worker (currently we are using buildbot version 2.8.4).
   This specific version can be installed using ``pip``, with a command such
   as ``pip3 install buildbot-worker==2.8.4``.

#. Create a designated user account, your buildbot-worker will be running under,
   and set appropriate permissions.

#. Choose the buildbot-worker root directory (all builds will be placed under
   it), buildbot-worker access name and password the build master will be using
   to authenticate your buildbot-worker.

#. Create a buildbot-worker in context of that buildbot-worker account. Point it
   to the **lab.llvm.org** port **9994** (see `Buildbot documentation,
   Creating a worker
   <http://docs.buildbot.net/current/tutorial/firstrun.html#creating-a-worker>`_
   for more details) by running the following command:

    .. code-block:: bash

       $ buildbot-worker create-worker <buildbot-worker-root-directory> \
                    lab.llvm.org:9994 \
                    <buildbot-worker-access-name> \
                    <buildbot-worker-access-password>

   Only once a new worker is stable, and
   approval from Galina has been received (see last step) should it
   be pointed at the main buildmaster.

   Now start the worker:

    .. code-block:: bash

       $ buildbot-worker start <buildbot-worker-root-directory>

   This will cause your new worker to connect to the staging buildmaster
   which is silent by default.

   Try this once then check the log file
   ``<buildbot-worker-root-directory>/worker/twistd.log``. If your settings
   are correct you will see a refused connection. This is good and expected,
   as the credentials have not been established on both ends. Now stop the
   worker and proceed to the next steps.

#. Fill the buildbot-worker description and admin name/e-mail.  Here is an
   example of the buildbot-worker description::

       Windows 7 x64
       Core i7 (2.66GHz), 16GB of RAM

       g++.exe (TDM-1 mingw32) 4.4.0
       GNU Binutils 2.19.1
       cmake version 2.8.4
       Microsoft(R) 32-bit C/C++ Optimizing Compiler Version 16.00.40219.01 for 80x86

   See `here <http://docs.buildbot.net/current/manual/installation/worker.html>`_
   for which files to edit.

#. Send a patch which adds your build worker and your builder to
   `zorg <https://github.com/llvm/llvm-zorg>`_. Use the typical LLVM
   `workflow <https://llvm.org/docs/Contributing.html#how-to-submit-a-patch>`_.

   * workers are added to ``buildbot/osuosl/master/config/workers.py``
   * builders are added to ``buildbot/osuosl/master/config/builders.py``

   Please make sure your builder name and its builddir are unique through the
   file.

   All new builders should default to using the "'collapseRequests': False"
   configuration.  This causes the builder to build each commit individually
   and not merge build requests.  To maximize quality of feedback to developers,
   we *strongly prefer* builders to be configured not to collapse requests.
   This flag should be removed only after all reasonable efforts have been
   exhausted to improve build times such that the builder can keep up with
   commit flow.

   It is possible to allow email addresses to unconditionally receive
   notifications on build failure; for this you'll need to add an
   ``InformativeMailNotifier`` to ``buildbot/osuosl/master/config/status.py``.
   This is particularly useful for the staging buildmaster which is silent
   otherwise.

#. Send the buildbot-worker access name and the access password directly to
   `Galina Kistanova <mailto:gkistanova@gmail.com>`_, and wait until she
   lets you know that your changes are applied and buildmaster is
   reconfigured.

#. Make sure you can start the buildbot-worker and successfully connect
   to the silent buildmaster. Then set up your buildbot-worker to start
   automatically at the start up time.  See the buildbot documentation
   for help.  You may want to restart your computer to see if it works.

#. Check the status of your buildbot-worker on the `Waterfall Display (Staging)
   <http://lab.llvm.org/staging/#/waterfall>`_ to make sure it is
   connected, and the `Workers Display (Staging)
   <http://lab.llvm.org/staging/#/workers>`_ to see if administrator
   contact and worker information are correct.

#. At this point, you have a working builder connected to the staging
   buildmaster.  You can now make sure it is reliably green and keeps
   up with the build queue.  No notifications will be sent, so you can
   keep an unstable builder connected to staging indefinitely.

#. (Optional) Once the builder is stable on the staging buildmaster with
   several days of green history, you can choose to move it to the production
   buildmaster to enable developer notifications.  Please email `Galina
   Kistanova <mailto:gkistanova@gmail.com>`_ for review and approval.

   To move a worker to production (once approved), stop your worker, edit the
   buildbot.tac file to change the port number from 9994 to 9990 and start it
   again.

Testing a Builder Config Locally
================================

It is possible to test a builder running against a local version of LLVM's
buildmaster setup. This allows you to test changes to builder, worker, and
buildmaster configuration. A buildmaster launched in this "local testing" mode
will:

* Bind only to local interfaces.
* Use SQLite as the database.
* Use a single fixed password for workers.
* Disable extras like GitHub authentication.

In order to use this "local testing" mode:

* Within a checkout of `llvm-zorg <https://github.com/llvm/llvm-zorg>`_,
  create and activate a Python `venv
  <https://docs.python.org/3/library/venv.html>`_ and install the necessary
  dependencies.

    .. code-block:: bash

       python -m venv bbenv
       source bbenv/bin/activate
       pip install buildbot{,-console-view,-grid-view,-waterfall-view,-worker,-www}==3.11.7 urllib3

* Initialise the necessary buildmaster files, link to the configuration in
  ``llvm-zorg`` and ask ``buildbot`` to check the configuration. This step can
  be run from any directory.

    .. code-block:: bash

       buildbot create-master llvm-testbbmaster
       cd llvm-testbbmaster
       ln -s /path/to/checkout/of/llvm-zorg/buildbot/osuosl/master/master.cfg .
       ln -s /path/to/checkout/of/llvm-zorg/buildbot/osuosl/master/config/ .
       ln -s /path/to/checkout/of/llvm-zorg/zorg/ .
       BUILDBOT_TEST=1 buildbot checkconfig

* Start the buildmaster.

    .. code-block:: bash

       BUILDBOT_TEST=1 buildbot start --nodaemon .

* After waiting a few seconds for startup to complete, you should be able to
  open the web UI at ``http://localhost:8011``.  If there are any errors or
  this isn't working, check ``twistd.log`` (within the current directory) for
  more information.

* You can now create and start a buildbot worker. Ensure you pick the correct
  name for the worker associated with the build configuration you want to test
  in ``buildbot/osuosl/master/config/builders.py``.

    .. code-block:: bash

       buildbot-worker create-worker <buildbot-worker-root-directory> \
                       localhost:9990 \
                       <buildbot-worker-name> \
                       test
       buildbot-worker start --nodaemon <buildbot-worker-root-directory>

* Either wait until the poller sets off a build, or alternatively force a
  build to start in the web UI.

* Review the progress and results of the build in the web UI.

This local testing configuration defaults to binding only to the loopback
interface for security reasons.

If you want to run the test worker on a different machine, or to run the
buildmaster on a remote server, ssh port forwarding can be used to make
connection possible. For instance, if running the buildmaster on a remote
server the following command will suffice to make the web UI accessible via
``http://localhost:8011`` and make it possible for a local worker to connect
to the remote buildmaster by connecting to ``localhost:9900``:

    .. code-block:: bash

       ssh -N -L 8011:localhost:8011 -L 9990:localhost:9990 username@buildmaster_server_address


Best Practices for Configuring a Fast Builder
=============================================

As mentioned above, we generally have a strong preference for
builders which can build every commit as they come in.  This section
includes best practices and some recommendations as to how to achieve
that end.

The goal
  In 2020, the monorepo had just under 35 thousand commits.  This works
  out to an average of 4 commits per hour.  Already, we can see that a
  builder must cycle in less than 15 minutes to have a hope of being
  useful.  However, those commits are not uniformly distributed.  They
  tend to cluster strongly during US working hours.  Looking at a couple
  of recent (Nov 2021) working days, we routinely see ~10 commits per
  hour during peek times, with occasional spikes as high as ~15 commits
  per hour.  Thus, as a rule of thumb, we should plan for our builder to
  complete ~10-15 builds an hour.

Resource Appropriately
  At 10-15 builds per hour, we need to complete a new build on average every
  4 to 6 minutes.  For anything except the fastest of hardware/build configs,
  this is going to be well beyond the ability of a single machine.  In buildbot
  terms, we likely going to need multiple workers to build requests in parallel
  under a single builder configuration.  For some rough back of the envelope
  numbers, if your build config takes e.g. 30 minutes, you will need something
  on the order of 5-8 workers.  If your build config takes ~2 hours, you'll
  need something on the order of 20-30 workers.  The rest of this section
  focuses on how to reduce cycle times.

Restrict what you build and test
  Think hard about why you're setting up a bot, and restrict your build
  configuration as much as you can.  Basic functionality is probably
  already covered by other bots, and you don't need to duplicate that
  testing.  You only need to be building and testing the *unique* parts
  of the configuration.  (e.g. For a multi-stage clang builder, you probably
  don't need to be enabling every target or building all the various utilities.)

  It can sometimes be worthwhile splitting a single builder into two or more,
  if you have multiple distinct purposes for the same builder.  As an example,
  if you want to both a) confirm that all of LLVM builds with your host
  compiler, and b) want to do a multi-stage clang build on your target, you
  may be better off with two separate bots.  Splitting increases resource
  consumption, but makes it easy for each bot to keep up with commit flow.
  Additionally, splitting bots may assist in triage by narrowing attention to
  relevant parts of the failing configuration.

  In general, we recommend Release build types with Assertions enabled.  This
  generally provides a good balance between build times and bug detection for
  most buildbots.  There may be room for including some debug info (e.g. with
  `-gmlt`), but in general the balance between debug info quality and build
  times is a delicate one.

Use Ninja & LLD
  Ninja really does help build times over Make, particularly for highly
  parallel builds.  LLD helps to reduce both link times and memory usage
  during linking significantly.  With a build machine with sufficient
  parallelism, link times tend to dominate critical path of the build, and are
  thus worth optimizing.

Use CCache and NOT incremental builds
  Using ccache materially improves average build times.  Incremental builds
  can be slightly faster, but introduce the risk of build corruption due to
  e.g. state changes, etc...  At this point, the recommendation is not to
  use incremental builds and instead use ccache as the latter captures the
  majority of the benefit with less risk of false positives.

  One of the non-obvious benefits of using ccache is that it makes the
  builder less sensitive to which projects are being monitored vs built.
  If a change triggers a build request, but doesn't change the build output
  (e.g. doc changes, python utility changes, etc..), the build will entirely
  hit in cache and the build request will complete in just the testing time.

  With multiple workers, it is tempting to try to configure a shared cache
  between the workers.  Experience to date indicates this is difficult to
  well, and that having local per-worker caches gets most of the benefit
  anyways.  We don't currently recommend shared caches.

  CCache does depend on the builder hardware having sufficient IO to access
  the cache with reasonable access times - i.e. a fast disk, or enough memory
  for a RAM cache, etc..  For builders without, incremental may be your best
  option, but is likely to require higher ongoing involvement from the
  sponsor.

Enable batch builds
  As a last resort, you can configure your builder to batch build requests.
  This makes the build failure notifications markedly less actionable, and
  should only be done once all other reasonable measures have been taken.

Leave it on the staging buildmaster
  While most of this section has been biased towards builders intended for
  the main buildmaster, it is worth highlighting that builders can run
  indefinitely on the staging buildmaster.  Such a builder may still be
  useful for the sponsoring organization, without concern of negatively
  impacting the broader community.  The sponsoring organization simply
  has to take on the responsibility of all bisection and triage.

Managing a Worker From The Web Interface
========================================

Tasks such as clearing pending building requests can be done using
the Buildbot web interface. To do this you must be recognised as an admin
of the worker:

* Set your public GitHub profile email to one that was included in the
  ``admin`` information you set up on the worker. It does not matter if this
  is your primary account email or a "verified email". To confirm this has been
  done correctly, go to ``github.com/<your GitHub username>`` and you should
  see the email address listed there.

  A worker can have many admins, if they are listed in the form
  ``First Last <first.last@example.com>, First2 Last2 <first2.last2@example.com>``.
  You only need to have one of those addresses in your profile to be recognised
  as an admin.

  If you need to add an email address, you can edit the ``admin`` file and
  restart the worker. You should see the new admin details in the web interface
  shortly afterwards.

* Connect GitHub to Buildbot by clicking on the "Anonymous" button on the
  top right of the page, then "Login with GitHub" and authorise the app.

Some tasks don't give immediate feedback, so if nothing happens within a short
time, try again with the browser's web console open. Sometimes you will see
403 errors and other messages that might indicate you don't have the correct
details set up.

