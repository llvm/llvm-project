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
It is located at ``llvm/Telemetry/Telemetry.h``

Characteristics
---------------
* Configurable and extensible by:

  * Tools: any tool that wants to use Telemetry can extend and customize it.
  * Vendors: Toolchain vendors can also provide custom implementation of the
    library, which could either override or extend the given tool's upstream
    implementation, to best fit their organization's usage and privacy models.
  * End users of such tool can also configure Telemetry (as allowed by their
    vendor).

Important notes
----------------

* There is no concrete implementation of a Telemetry library in upstream LLVM.
  We only provide the abstract API here. Any tool that wants telemetry will
  implement one.
  
  The rationale for this is that all the tools in LLVM are very different in
  what they care about (what/where/when to instrument data). Hence, it might not
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
  
    * E.g., data collected by Telemetry is not necessarily owned by the user
      of an LLVM tool with Telemetry enabled, hence the user's consent to data
      collection is not meaningful. On the other hand, LLVM developers have no
      reasonable ways to request consent from the "real" owners.


High-level design
=================

Key components
--------------

The framework consists of four important classes:

* ``llvm::telemetry::Manager``: The class responsible for collecting and
  transmitting telemetry data. This is the main point of interaction between the
  framework and any tool that wants to enable telemetry.
* ``llvm::telemetry::TelemetryInfo``: Data courier
* ``llvm::telemetry::Destination``: Data sink to which the Telemetry framework
  sends data.
  Its implementation is transparent to the framework.
  It is up to the vendor to decide which pieces of data to forward and where
  to forward them to for their final storage.
* ``llvm::telemetry::Config``: Configurations for the ``Manager``.
  
.. image:: llvm_telemetry_design.png

How to implement and interact with the API
------------------------------------------

To use Telemetry in your tool, you need to provide a concrete implementation of the ``Manager`` class and ``Destination``.

1) Define a custom ``Serializer``, ``Manager``, ``Destination`` and optionally a subclass of ``TelemetryInfo``

.. code-block:: c++

  class JsonSerializer : public Serializer {
  public:
    json::Object *getOutputObject() { return object.get(); }

    llvm::Error init() override {
      if (started)
        return createStringError("Serializer already in use");
      started = true;
      object = std::make_unique<json::Object>();
      return Error::success();
    }

    // Serialize the given value.
    void write(StringRef KeyName, bool Value) override {
      writeHelper(KeyName, Value);
    }

    void write(StringRef KeyName, int Value) override {
      writeHelper(KeyName, Value);
    }

    void write(StringRef KeyName, size_t Value) override {
      writeHelper(KeyName, Value);
    }
    void write(StringRef KeyName, StringRef Value) override {
      writeHelper(KeyName, Value);
    }

    void write(StringRef KeyName,
               const std::map<std::string, std::string>& Value) override {
      json::Object Inner;
      for (auto kv : Value) {
        Inner.try_emplace(kv.first, kv.second);
      }
      writeHelper(KeyName, json::Value(std::move(Inner)));
    }

    Error finalize() override {
      if (!started)
        return createStringError("Serializer not currently in use");
      started = false;
      return Error::success();
    }

  private:
    template <typename T> void writeHelper(StringRef Name, T Value) {
      assert(started && "serializer not started");
      object->try_emplace(Name, Value);
    }
    bool started = false;
    std::unique_ptr<json::Object> object;
  };
       
  class MyManager : public telemery::Manager {
  public:
  static std::unique_ptr<MyManager> createInstatnce(telemetry::Config* config) {
    // If Telemetry is not enabled, then just return null;
    if (!config->EnableTelemetry) return nullptr;

    return std::make_unique<MyManager>();
  }
  MyManager() = default;

  Error dispatch(TelemetryInfo* Entry) const override {
    Entry->SessionId = SessionId;
    emitToAllDestinations(Entry);
  }
      
  void addDestination(std::unique_ptr<Destination> dest) override {
    destinations.push_back(std::move(dest));
  }
  
  // You can also define additional instrumentation points.
  void logStartup(TelemetryInfo* Entry) {
    // Add some additional data to entry.
    Entry->Msg = "Some message";
    dispatch(Entry);
  }
  
  void logAdditionalPoint(TelemetryInfo* Entry) {
    // .... code here
  }
  
  private:
    void emitToAllDestinations(const TelemetryInfo* Entry) {
      for (Destination* Dest : Destinations) {
        Dest->receiveEntry(Entry);
      }
    }
    
    std::vector<Destination> Destinations;
    const std::string SessionId;
  };

  class MyDestination : public telemetry::Destination {
  public:
    Error receiveEntry(const TelemetryInfo* Entry) override {
      if (Error err = serializer.init()) {
        return err;
      }
      Entry->serialize(serializer);
      if (Error err = serializer.finalize()) {
        return err;
      }

      json::Object copied = *serializer.getOutputObject();
      // Send the `copied` object to wherever.
      return Error::success();
    }

  private:
    JsonSerializer serializer;
  };

  // This defines a custom TelemetryInfo that has an additional Msg field.
  struct MyTelemetryInfo : public telemetry::TelemetryInfo {
    std::string Msg;
    
    Error serialize(Serializer& serializer) const override {
      TelemetryInfo::serialize(serializer);
      serializer.writeString("MyMsg", Msg);
    }
      
    // Note: implement getKind() and classof() to support dyn_cast operations.
  };

    
2) Use the library in your tool.

Logging the tool init-process:

.. code-block:: c++

  // In tool's initialization code.
  auto StartTime = std::chrono::time_point<std::chrono::steady_clock>::now();
  telemetry::Config MyConfig = makeConfig(); // Build up the appropriate Config struct here.
  auto Manager = MyManager::createInstance(&MyConfig);

  
  // Any other tool's init code can go here
  // ...
  
  // Finally, take a snapshot of the time now so we know how long it took the
  // init process to finish
  auto EndTime = std::chrono::time_point<std::chrono::steady_clock>::now();
  MyTelemetryInfo Entry;

  Entry.Start = StartTime;
  Entry.End = EndTime;
  Manager->logStartup(&Entry);

Similar code can be used for logging the tool's exit.

