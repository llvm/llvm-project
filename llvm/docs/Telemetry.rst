===========================
Telemetry framework in LLVM
===========================

.. contents::
   :local:

.. toctree::
   :hidden:

===========================
Telemetry framework in LLVM
===========================

Objective
=========

Provides a common framework in LLVM for collecting various usage and performance
metrics.
It is located at `llvm/Telemetry/Telemetry.h`

Characteristics
---------------
* Configurable and extensible by:

  * Tools: any tool that wants to use Telemetry can extend and customize it.
  * Vendors: Toolchain vendors can also provide custom implementation of the
    library, which could either override or extend the given tool's upstream
    implementation, to best fit their organization's usage and privacy models.
  * End users of such tool can also configure Telemetry(as allowed by their
    vendor).


Important notes
----------------

* There is no concrete implementation of a Telemetry library in upstream LLVM.
  We only provide the abstract API here. Any tool that wants telemetry will
  implement one.
  
  The rationale for this is that, all the tools in llvm are very different in
  what they care about(what/where/when to instrument data). Hence, it might not
  be practical to have a single implementation.
  However, in the future, if we see enough common pattern, we can extract them
  into a shared place. This is TBD - contributions are welcomed.

* No implementation of Telemetry in upstream LLVM shall store any of the
  collected data due to privacy and security reasons:
  
  * Different organizations have different privacy models:
  
    * Which data is sensitive, which is not?
    * Whether it is acceptable for instrumented data to be stored anywhere?
      (to a local file, what not?)
      
  * Data ownership and data collection consents are hard to accommodate from
    LLVM developers' point of view:
     * Eg., data collected by Telemetry is not neccessarily owned by the user
       of an LLVM tool with Telemetry enabled, hence the user's consent to data
       collection is not meaningful. On the other hand, LLVM developers have no
       reasonable ways to request consent from the "real" owners.


High-level design
=================

Key components
--------------

The framework is consisted of three important classes:

* `llvm::telemetry::Telemeter`: The class responsible for collecting and
  forwarding telemetry data. This is the main point of interaction between the
  framework and any tool that wants to enable telemery.
* `llvm::telemetry::TelemetryInfo`: Data courier
* `llvm::telemetry::Config`: Configurations on the `Telemeter`.

.. image:: llvm_telemetry_design.png
